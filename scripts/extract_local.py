"""
scripts/extract_local.py
------------------------
Local replacement for the CDSE openEO point-extraction pipeline.

Produces ``.geoparquet`` files with the exact schema CDSE writes (verified
column-by-column against ``2025_uzcosmos_finetune_v2_w0`` outputs), so the
existing ``merge`` step in ``extract_finetune_points.py`` and
``worldcereal.utils.timeseries.process_parquet`` consume them unchanged.

Why local
~~~~~~~~~
- CDSE Spark workers OOM repeatedly on dense tiles (42TUK is the headline).
- No CDSE quota; runs on whatever compute you have.
- Reproducible and step-debuggable.

Scope
~~~~~
Full CDSE-equivalent feature set:

- **S2-L2A**: monthly median composites with SCL cloud mask (Planetary Computer).
- **S1-RTC**: monthly mean already-terrain-corrected sigma0 (Planetary Computer).
- **DEM**: Cop-DEM-GLO-30 elevation + slope computed from elevation gradient.
- **AgERA5**: monthly mean temperature + monthly mean daily precipitation_flux,
  fetched via Open-Meteo Historical Weather API (ERA5-derived, no auth).

All four families can be individually skipped via flags if the network/source
is unavailable; missing values are filled with ``NODATAVALUE = 65535`` which
``process_parquet`` tolerates.

Usage
~~~~~
    python scripts/extract_local.py extract \\
        --samples outputs/finetune/samples_worldcereal_w0.geoparquet \\
        --ref-id  2025_uzcosmos_finetune_v2_w0_LOCAL \\
        --start-date 2024-08-01 --end-date 2026-03-31 \\
        --tile 42TUK
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable

# GDAL/curl tuning for COG range-reads. Must be set BEFORE rasterio imports.
# - HTTP/2 multiplexing: many concurrent reads share one TLS connection.
# - VSI_CACHE: in-process LRU keeps recently-fetched COG headers / blocks
#   so repeated band reads of the same scene reuse already-fetched bytes.
# - DISABLE_READDIR_ON_OPEN: skips a ``LIST`` call that's pointless on
#   single-COG buckets and adds 100-200 ms per asset on Azure blob storage.
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("VSI_CACHE", "TRUE")
os.environ.setdefault("VSI_CACHE_SIZE", "536870912")  # 512 MiB
os.environ.setdefault("GDAL_HTTP_MULTIPLEX", "YES")
os.environ.setdefault("GDAL_HTTP_VERSION", "2")
os.environ.setdefault("CPL_VSIL_CURL_USE_HEAD", "NO")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "3")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "1")

import click
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rioxarray  # noqa: F401  (registers the .rio accessor)
import xarray as xr
from pystac_client import Client
from shapely.geometry import box, mapping

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
_log = logging.getLogger("extract_local")

ROOT = Path(__file__).resolve().parents[1]
EXTRACTIONS_ROOT = ROOT / "outputs" / "finetune" / "extractions"

# uint16 NODATA used everywhere downstream.
NODATAVALUE = 65535

# Schema constants — must match CDSE.
S2_BAND_NAMES = [
    "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12",
]
S2_PARQUET_COLS = [f"S2-L2A-{b}" for b in S2_BAND_NAMES]
S1_PARQUET_COLS = ["S1-SIGMA0-VH", "S1-SIGMA0-VV"]

# Cloud-mask SCL classes to drop (cloud / shadow / cirrus / saturated).
SCL_DROP = {3, 8, 9, 10, 11}

PC_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"


def _l2a_boa_offset(item) -> int:
    """Return the BOA_ADD_OFFSET magnitude for an S2-L2A STAC item.

    Items processed with baseline ≥ 04.00 (introduced 2022-01-25) carry an
    additive offset of −1000 in the surface reflectance bands. Microsoft
    Planetary Computer serves the **raw DN with the offset still embedded**,
    while the WorldCereal training contract (CDSE openEO `SENTINEL2_L2A`)
    expects the offset already applied. Without subtracting this magnitude
    every fine-tune sample lands ~+1000 above the pretrained encoder's
    distribution; the head learned at fine-tune time then maps "encoder-
    confused features → labels", and inference on offset-corrected cubes
    silently produces wrong predictions.

    Pre-baseline scenes (2018–2021) → 0. SCL is not affected by the offset.
    """
    baseline = item.properties.get("s2:processing_baseline")
    try:
        return 1000 if baseline is not None and float(baseline) >= 4.00 else 0
    except (TypeError, ValueError):
        return 0


def _sigma0_linear_to_uint16(arr: np.ndarray) -> np.ndarray:
    """Match the CDSE encoding used in extracted parquets:
        uint16_val = 10^((10*log10(sigma0_linear) + 83) / 20)
    so the on-disk uint16 round-trips back to sigma0(dB) via
        dB = 20*log10(uint16) - 83
    which is what worldcereal.train.predictors expects.
    """
    out = np.full_like(arr, NODATAVALUE, dtype=np.uint16)
    valid = (arr > 0) & np.isfinite(arr)
    if not np.any(valid):
        return out
    db = 10.0 * np.log10(arr[valid])
    encoded = np.power(10.0, (db + 83.0) / 20.0)
    out[valid] = np.clip(encoded, 1, 65534).astype(np.uint16)
    return out


def _open_clipped(asset_href: str, bbox_4326: tuple[float, float, float, float]) -> xr.DataArray:
    """Open a signed PC STAC asset and clip to the AOI bbox during read.

    The clip is essential — without it, each ``open_rasterio + .values`` call
    pulls the full ~120 MB COG. With ``rio.clip_box`` rasterio issues a
    windowed range request and we transfer only a few KB per scene.
    """
    da = rioxarray.open_rasterio(asset_href, masked=True)
    return da.rio.clip_box(*bbox_4326, crs="EPSG:4326")


def _stac_search(
    *,
    collection: str,
    bbox: tuple[float, float, float, float],
    start: str,
    end: str,
    extra_query: dict | None = None,
):
    cat = Client.open(PC_STAC_URL)
    search = cat.search(
        collections=[collection],
        bbox=bbox,
        datetime=f"{start}/{end}",
        query=extra_query or {},
    )
    return list(search.items())


def _months_between(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    """Inclusive monthly index, anchored to month-start, between start and end."""
    return list(pd.date_range(start.to_period("M").to_timestamp(),
                              end.to_period("M").to_timestamp(),
                              freq="MS"))


def _signed_href(item, asset_key: str) -> str:
    """Sign a Planetary Computer asset URL for read access."""
    import planetary_computer as pc  # noqa: PLC0415
    asset = item.assets[asset_key]
    return pc.sign(asset.href)


def _bbox_from_samples(samples: gpd.GeoDataFrame, pad_m: float = 200.0) -> tuple[float, float, float, float]:
    """WGS84 bbox padded by ~200 m so we always have a pixel of headroom."""
    utm = samples.estimate_utm_crs()
    b = samples.to_crs(utm).total_bounds
    padded = box(b[0] - pad_m, b[1] - pad_m, b[2] + pad_m, b[3] + pad_m)
    minx, miny, maxx, maxy = gpd.GeoSeries([padded], crs=utm).to_crs("EPSG:4326").total_bounds
    return float(minx), float(miny), float(maxx), float(maxy)


def _sample_points(
    cube: xr.DataArray,
    points_xy: np.ndarray,
    src_crs: str,
) -> np.ndarray:
    """Sample ``cube`` (dims ``y, x``) at ``points_xy`` (Nx2 in EPSG:4326).

    Returns a length-N array, NaN for points outside the cube extent.
    Uses nearest-neighbour pixel centre.
    """
    from pyproj import Transformer  # noqa: PLC0415

    transformer = Transformer.from_crs("EPSG:4326", src_crs, always_xy=True)
    xs, ys = transformer.transform(points_xy[:, 0], points_xy[:, 1])
    return cube.sel(x=xr.DataArray(xs, dims="pt"),
                    y=xr.DataArray(ys, dims="pt"),
                    method="nearest").values


# ---------------------------------------------------------------------------
# S2 monthly composites
# ---------------------------------------------------------------------------

def _build_s2_monthly(
    bbox: tuple[float, float, float, float],
    months: list[pd.Timestamp],
    cloud_cover_max: int = 80,
    max_scenes_per_month: int = 4,
) -> dict[pd.Timestamp, dict[str, xr.DataArray]]:
    """All COG reads are clipped to ``bbox`` (WGS84) — see ``_open_clipped``."""
    """Returns {month_ts: {band_name: 2D DataArray}} — uint16 DN scale.

    Stacks the ``max_scenes_per_month`` cleanest scenes per month (sorted by
    ``eo:cloud_cover`` ASC), applies SCL cloud mask, takes median per band.
    Skips a month if there are zero clear pixels.

    Why a scene cap: monthly median converges quickly — 4 cleanest scenes
    deliver virtually the same composite as 20+, while cutting download
    time ~5×. Matches sentinelhub's ``_WC_MAX_SCENES_PER_MONTH_S2``.
    """
    items_all = _stac_search(
        collection="sentinel-2-l2a",
        bbox=bbox,
        start=str(months[0].date()),
        end=str((months[-1] + pd.offsets.MonthEnd(0)).date()),
        extra_query={"eo:cloud_cover": {"lt": cloud_cover_max}},
    )
    _log.info("S2 STAC: %d items in bbox %s", len(items_all), bbox)

    by_month: dict[pd.Timestamp, list] = {m: [] for m in months}
    for it in items_all:
        ts = pd.to_datetime(it.properties["datetime"]).tz_localize(None)
        m_key = pd.Timestamp(ts.year, ts.month, 1)
        if m_key in by_month:
            by_month[m_key].append(it)

    out: dict[pd.Timestamp, dict[str, xr.DataArray]] = {}
    for m in months:
        items = by_month.get(m, [])
        if not items:
            _log.warning("S2 month %s: no scenes", m.date())
            continue
        # Cap: keep the cleanest N scenes by eo:cloud_cover.
        if max_scenes_per_month and len(items) > max_scenes_per_month:
            items = sorted(
                items,
                key=lambda it: it.properties.get("eo:cloud_cover", 100.0),
            )[:max_scenes_per_month]
        _log.info("S2 month %s: %d scenes (cap=%d)",
                  m.date(), len(items), max_scenes_per_month)

        scene_stacks: list[xr.DataArray] = []
        for it in items:
            try:
                offset = _l2a_boa_offset(it)
                scl = _open_clipped(_signed_href(it, "SCL"), bbox).squeeze("band")
                cloud = scl.isin(list(SCL_DROP))
                bands = []
                for b in S2_BAND_NAMES:
                    da = _open_clipped(_signed_href(it, b), bbox).squeeze("band")
                    da = da.rio.reproject_match(scl)
                    if offset:
                        # Subtract baseline-04.00 BOA_ADD_OFFSET, clamp at 0,
                        # then re-apply the cloud mask. The order matters: if
                        # we masked first the offset would shift NODATAVALUE
                        # itself by −1000, breaking downstream comparisons.
                        da = xr.where(
                            da == NODATAVALUE, NODATAVALUE,
                            xr.where(da > offset, da - offset, 0),
                        )
                    da = da.where(~cloud, NODATAVALUE)
                    bands.append(da.expand_dims(band=[b]))
                scene_stacks.append(xr.concat(bands, dim="band"))
            except Exception as exc:  # noqa: BLE001
                _log.warning("S2 scene %s skipped: %s", it.id, exc)

        if not scene_stacks:
            continue

        stack = xr.concat(scene_stacks, dim="scene")
        # Median ignoring nodata.
        masked = stack.where(stack != NODATAVALUE)
        composite = masked.median(dim="scene", skipna=True)
        composite = composite.fillna(NODATAVALUE).astype(np.uint16)

        out[m] = {b: composite.sel(band=b) for b in S2_BAND_NAMES}
    return out


# ---------------------------------------------------------------------------
# S1-RTC monthly mean
# ---------------------------------------------------------------------------

def _select_best_s1_orbit(
    bbox: tuple[float, float, float, float],
    start: str,
    end: str,
) -> str | None:
    """Pick the S1 orbit direction with more items overlapping *bbox* across
    the year window. Mirrors training's ``select_best_s1_orbit_direction``
    in ``worldcereal/openeo/preprocessing.py`` so the encoder sees the same
    look-angle pattern at fine-tune time as at inference.
    """
    counts: dict[str, int] = {}
    for orbit in ("ascending", "descending"):
        items = _stac_search(
            collection="sentinel-1-rtc",
            bbox=bbox,
            start=start,
            end=end,
            extra_query={"sat:orbit_state": {"eq": orbit}},
        )
        counts[orbit] = len(items)
    _log.info("S1 orbit counts over %s..%s: %s", start, end, counts)
    if not any(counts.values()):
        return None
    return max(counts, key=counts.get)


def _build_s1_monthly(
    bbox: tuple[float, float, float, float],
    months: list[pd.Timestamp],
    max_scenes_per_month: int = 6,
) -> dict[pd.Timestamp, dict[str, xr.DataArray]]:
    """S1-RTC is already terrain-corrected sigma0 (linear). Monthly mean →
    uint16 sigma0-encoded.

    Filters to a single best-overlap orbit direction so backscatter texture
    matches what training expects. Cap: keeps the chronologically first
    ``max_scenes_per_month`` scenes per month (S1 has no cloud-cover;
    relative-orbit / acquisition-date spread is already implicit).
    """
    start = str(months[0].date())
    end = str((months[-1] + pd.offsets.MonthEnd(0)).date())

    s1_orbit = _select_best_s1_orbit(bbox, start, end)
    _log.info("S1 orbit selected: %s", s1_orbit)

    extra_query: dict | None = None
    if s1_orbit:
        extra_query = {"sat:orbit_state": {"eq": s1_orbit}}

    items_all = _stac_search(
        collection="sentinel-1-rtc",
        bbox=bbox,
        start=start,
        end=end,
        extra_query=extra_query,
    )
    _log.info("S1-RTC STAC: %d items in bbox %s", len(items_all), bbox)

    by_month: dict[pd.Timestamp, list] = {m: [] for m in months}
    for it in items_all:
        ts = pd.to_datetime(it.properties["datetime"]).tz_localize(None)
        m_key = pd.Timestamp(ts.year, ts.month, 1)
        if m_key in by_month:
            by_month[m_key].append(it)

    out: dict[pd.Timestamp, dict[str, xr.DataArray]] = {}
    for m in months:
        items = by_month.get(m, [])
        if not items:
            _log.warning("S1 month %s: no scenes", m.date())
            continue
        if max_scenes_per_month and len(items) > max_scenes_per_month:
            items = sorted(
                items,
                key=lambda it: it.properties.get("datetime", ""),
            )[:max_scenes_per_month]
        _log.info("S1 month %s: %d scenes (cap=%d)",
                  m.date(), len(items), max_scenes_per_month)

        per_pol: dict[str, list] = {"vh": [], "vv": []}
        ref = None
        for it in items:
            try:
                vh = _open_clipped(_signed_href(it, "vh"), bbox).squeeze("band")
                vv = _open_clipped(_signed_href(it, "vv"), bbox).squeeze("band")
                if ref is None:
                    ref = vh
                vh = vh.rio.reproject_match(ref)
                vv = vv.rio.reproject_match(ref)
                per_pol["vh"].append(vh)
                per_pol["vv"].append(vv)
            except Exception as exc:  # noqa: BLE001
                _log.warning("S1 scene %s skipped: %s", it.id, exc)

        if not per_pol["vh"]:
            continue

        out_bands: dict[str, xr.DataArray] = {}
        for pol_key, parquet_col in (("vh", "S1-SIGMA0-VH"), ("vv", "S1-SIGMA0-VV")):
            stack = xr.concat(per_pol[pol_key], dim="scene")
            mean_lin = stack.mean(dim="scene", skipna=True)
            encoded = xr.apply_ufunc(
                _sigma0_linear_to_uint16,
                mean_lin,
                dask="parallelized",
                output_dtypes=[np.uint16],
            )
            out_bands[parquet_col] = encoded
        out[m] = out_bands
    return out


# ---------------------------------------------------------------------------
# Cop-DEM-GLO-30 → static elevation + slope per sample
# ---------------------------------------------------------------------------

def _build_dem_static(
    samples: gpd.GeoDataFrame,
    bbox: tuple[float, float, float, float],
) -> dict[str, np.ndarray]:
    """Return ``{"elevation": [...], "slope": [...]}`` (uint16 per sample).

    - elevation: metres above sea level, clipped to uint16 range.
    - slope: degrees, computed from elevation gradient on the local UTM grid
      so dx/dy are metres (not degrees), then ``arctan(|grad|)`` in degrees.
    - both values are nearest-neighbour sampled at the sample centroids.
    """
    items = _stac_search(
        collection="cop-dem-glo-30",
        bbox=bbox,
        # cop-dem is static — a wide window guarantees all candidate tiles.
        start="2010-01-01",
        end="2030-12-31",
    )
    if not items:
        _log.warning("Cop-DEM-GLO-30: no items for bbox %s", bbox)
        return {
            "elevation": np.full(len(samples), NODATAVALUE, dtype=np.uint16),
            "slope": np.full(len(samples), NODATAVALUE, dtype=np.uint16),
        }
    _log.info("Cop-DEM-GLO-30: %d tile(s)", len(items))

    # Open + clip each 1° tile, then mosaic via ``rio.merge``.
    from rioxarray.merge import merge_arrays  # noqa: PLC0415

    clipped: list[xr.DataArray] = []
    for it in items:
        try:
            da = _open_clipped(_signed_href(it, "data"), bbox).squeeze("band")
            if "band" in da.dims:
                da = da.squeeze("band", drop=True)
            clipped.append(da)
        except Exception as exc:  # noqa: BLE001
            _log.warning("DEM tile %s skipped: %s", it.id, exc)

    if not clipped:
        return {
            "elevation": np.full(len(samples), NODATAVALUE, dtype=np.uint16),
            "slope": np.full(len(samples), NODATAVALUE, dtype=np.uint16),
        }

    if len(clipped) == 1:
        elev = clipped[0]
    else:
        elev = merge_arrays(clipped)
        if "band" in elev.dims:
            elev = elev.squeeze("band", drop=True)

    # Reproject to local UTM so slope is computed in metres.
    utm_crs = samples.estimate_utm_crs()
    elev_utm = elev.rio.reproject(str(utm_crs))

    arr = elev_utm.values.astype(np.float32)
    # Pixel size in metres (rioxarray's transform exposes both).
    tr = elev_utm.rio.transform()
    px = abs(tr.a)
    py = abs(tr.e)
    gy, gx = np.gradient(arr, py, px)
    slope_deg = np.degrees(np.arctan(np.sqrt(gx**2 + gy**2)))

    slope_da = xr.DataArray(
        slope_deg,
        dims=elev_utm.dims,
        coords=elev_utm.coords,
    ).rio.write_crs(elev_utm.rio.crs)

    pts_lonlat = np.column_stack(
        [samples.geometry.x.values, samples.geometry.y.values]
    )
    elev_vals = _sample_points(elev_utm, pts_lonlat, src_crs=str(utm_crs))
    slope_vals = _sample_points(slope_da, pts_lonlat, src_crs=str(utm_crs))

    elev_u16 = np.where(np.isnan(elev_vals), NODATAVALUE,
                        np.clip(elev_vals, 0, 65534)).astype(np.uint16)
    slope_u16 = np.where(np.isnan(slope_vals), NODATAVALUE,
                         np.clip(slope_vals, 0, 65534)).astype(np.uint16)
    return {"elevation": elev_u16, "slope": slope_u16}


# ---------------------------------------------------------------------------
# AgERA5 monthly via Open-Meteo Historical Weather API (ERA5-derived)
# ---------------------------------------------------------------------------

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"


def _snap_to_grid(lat: float, lon: float, step: float = 0.1) -> tuple[float, float]:
    """Round to the nearest 0.1° cell (AgERA5 native resolution)."""
    return round(lat / step) * step, round(lon / step) * step


def _fetch_open_meteo_daily(
    lat: float, lon: float, start: str, end: str,
) -> pd.DataFrame:
    """One HTTP call → DataFrame indexed by date with mean temp (K) and
    daily precipitation (mm). Empty DataFrame on failure.
    """
    import urllib.request  # noqa: PLC0415
    import json as _json  # noqa: PLC0415

    params = {
        "latitude": f"{lat:.4f}",
        "longitude": f"{lon:.4f}",
        "start_date": start,
        "end_date": end,
        "daily": "temperature_2m_mean,precipitation_sum",
        "timezone": "UTC",
    }
    url = OPEN_METEO_URL + "?" + "&".join(f"{k}={v}" for k, v in params.items())
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            data = _json.loads(resp.read())
    except Exception as exc:  # noqa: BLE001
        _log.warning("Open-Meteo fetch failed (%.4f, %.4f): %s", lat, lon, exc)
        return pd.DataFrame()
    daily = data.get("daily") or {}
    if not daily.get("time"):
        return pd.DataFrame()
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(daily["time"]),
            "tmean_c": daily["temperature_2m_mean"],
            "precip_mm": daily["precipitation_sum"],
        }
    ).set_index("date")
    return df


def _build_agera5_monthly(
    samples: gpd.GeoDataFrame,
    months: list[pd.Timestamp],
    start_date: str,
    end_date: str,
) -> dict[pd.Timestamp, dict[str, np.ndarray]]:
    """Return ``{month: {"AGERA5-PRECIP": arr, "AGERA5-TMEAN": arr}}``.

    Encoding (matches CDSE parquets, decoded later by
    ``worldcereal.train.predictors``):
        AGERA5-TMEAN  = round(K * 100)        (e.g. 286.98 K → 28698)
        AGERA5-PRECIP = round(mm_per_day * 100) (e.g. 1.85 mm/day → 185)
    """
    n = len(samples)
    lats = samples.geometry.y.values
    lons = samples.geometry.x.values

    snapped = [_snap_to_grid(la, lo) for la, lo in zip(lats, lons)]
    sample_to_cell = np.array(snapped)
    unique_cells = sorted({tuple(c) for c in snapped})
    _log.info("AgERA5: %d sample(s) -> %d unique 0.1° cell(s)", n, len(unique_cells))

    cell_daily: dict[tuple[float, float], pd.DataFrame] = {}
    for la, lo in unique_cells:
        cell_daily[(la, lo)] = _fetch_open_meteo_daily(la, lo, start_date, end_date)

    # For each month: per-cell aggregation, then broadcast back to samples.
    out: dict[pd.Timestamp, dict[str, np.ndarray]] = {}
    for m in months:
        m_start = m
        m_end = m + pd.offsets.MonthEnd(0)
        cell_tmean: dict[tuple[float, float], float] = {}
        cell_precip: dict[tuple[float, float], float] = {}
        for cell, df in cell_daily.items():
            if df.empty:
                continue
            sub = df.loc[(df.index >= m_start) & (df.index <= m_end)]
            if sub.empty:
                continue
            tmean_c = sub["tmean_c"].mean(skipna=True)
            precip_mm_per_day = sub["precip_mm"].mean(skipna=True)
            if pd.notna(tmean_c):
                cell_tmean[cell] = tmean_c
            if pd.notna(precip_mm_per_day):
                cell_precip[cell] = precip_mm_per_day

        tmean_arr = np.full(n, NODATAVALUE, dtype=np.uint16)
        precip_arr = np.full(n, NODATAVALUE, dtype=np.uint16)
        for i in range(n):
            cell = (sample_to_cell[i, 0], sample_to_cell[i, 1])
            if cell in cell_tmean:
                k_x_100 = (cell_tmean[cell] + 273.15) * 100.0
                tmean_arr[i] = int(np.clip(round(k_x_100), 1, 65534))
            if cell in cell_precip:
                p_x_100 = cell_precip[cell] * 100.0
                precip_arr[i] = int(np.clip(round(p_x_100), 0, 65534))
        out[m] = {"AGERA5-TMEAN": tmean_arr, "AGERA5-PRECIP": precip_arr}

    return out


# ---------------------------------------------------------------------------
# Build the wide-format parquet
# ---------------------------------------------------------------------------

def _build_rows(
    samples: gpd.GeoDataFrame,
    s2_by_month: dict,
    s1_by_month: dict,
    months: list[pd.Timestamp],
    *,
    tile: str,
    ref_id: str,
    start_date: str,
    end_date: str,
    year: int,
    dem_static: dict[str, np.ndarray] | None = None,
    agera5_by_month: dict[pd.Timestamp, dict[str, np.ndarray]] | None = None,
) -> pd.DataFrame:
    """One row per (sample, month). Columns match CDSE wide-format schema."""
    pts_lonlat = np.column_stack([samples.geometry.x.values, samples.geometry.y.values])
    n_samples = len(samples)
    sample_ids = samples["sample_id"].astype(str).values

    rows: list[dict] = []
    for fi, m in enumerate(months):
        # Sample S2 bands
        s2_vals: dict[str, np.ndarray] = {}
        if m in s2_by_month:
            for b, parquet_col in zip(S2_BAND_NAMES, S2_PARQUET_COLS):
                da = s2_by_month[m][b]
                vals = _sample_points(da, pts_lonlat, src_crs=str(da.rio.crs))
                vals = np.where(np.isnan(vals), NODATAVALUE, vals).astype(np.uint16)
                s2_vals[parquet_col] = vals
        else:
            for parquet_col in S2_PARQUET_COLS:
                s2_vals[parquet_col] = np.full(n_samples, NODATAVALUE, dtype=np.uint16)

        # Sample S1 bands
        s1_vals: dict[str, np.ndarray] = {}
        if m in s1_by_month:
            for parquet_col in S1_PARQUET_COLS:
                da = s1_by_month[m][parquet_col]
                vals = _sample_points(da, pts_lonlat, src_crs=str(da.rio.crs))
                vals = np.where(np.isnan(vals), NODATAVALUE, vals).astype(np.uint16)
                s1_vals[parquet_col] = vals
        else:
            for parquet_col in S1_PARQUET_COLS:
                s1_vals[parquet_col] = np.full(n_samples, NODATAVALUE, dtype=np.uint16)

        # Per-month AgERA5 (uint16 already encoded).
        if agera5_by_month and m in agera5_by_month:
            tmean_month = agera5_by_month[m]["AGERA5-TMEAN"]
            precip_month = agera5_by_month[m]["AGERA5-PRECIP"]
        else:
            tmean_month = np.full(n_samples, NODATAVALUE, dtype=np.uint16)
            precip_month = np.full(n_samples, NODATAVALUE, dtype=np.uint16)

        for i in range(n_samples):
            row: dict = {
                "feature_index": fi,
                "sample_id": sample_ids[i],
                "timestamp": m,
                **{c: int(s2_vals[c][i]) for c in S2_PARQUET_COLS},
                **{c: int(s1_vals[c][i]) for c in S1_PARQUET_COLS},
                "slope": int(dem_static["slope"][i]) if dem_static is not None else NODATAVALUE,
                "elevation": int(dem_static["elevation"][i]) if dem_static is not None else NODATAVALUE,
                "AGERA5-PRECIP": int(precip_month[i]),
                "AGERA5-TMEAN": int(tmean_month[i]),
                "lon": float(pts_lonlat[i, 0]),
                "lat": float(pts_lonlat[i, 1]),
                "geometry": samples.geometry.iloc[i],
                "tile": tile,
                "h3_l3_cell": "",
                "start_date": start_date,
                "end_date": end_date,
                "year": int(year),
                "valid_time": str(samples["valid_time"].iloc[i]),
                "ewoc_code": int(samples["ewoc_code"].iloc[i]),
                "irrigation_status": int(samples.get("irrigation_status", pd.Series(0, index=samples.index)).iloc[i]),
                "quality_score_lc": int(samples.get("quality_score_lc", pd.Series(100, index=samples.index)).iloc[i]),
                "quality_score_ct": int(samples.get("quality_score_ct", pd.Series(100, index=samples.index)).iloc[i]),
                "extract": int(samples.get("extract", pd.Series(1, index=samples.index)).iloc[i]),
                "ref_id": ref_id,
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    # Match CDSE's datetime64[ns] precision exactly — process_parquet's dtype
    # checks reject the [us] default that newer pandas writes.
    df["timestamp"] = pd.to_datetime(df["timestamp"]).astype("datetime64[ns]")
    # Match CDSE dtypes — uint16 for the EO bands, int64 for metadata.
    for c in S2_PARQUET_COLS + S1_PARQUET_COLS + ["slope", "elevation", "AGERA5-PRECIP", "AGERA5-TMEAN"]:
        df[c] = df[c].astype("uint16")
    df["ref_id"] = df["ref_id"].astype("category")
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.group()
def cli() -> None:
    """Local extractor producing CDSE-compatible geoparquets."""


@cli.command("extract")
@click.option("--samples", "samples_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--ref-id", required=True, help="ref_id label written into the parquet and used in the output dir name.")
@click.option("--start-date", required=True, help="YYYY-MM-DD inclusive.")
@click.option("--end-date", required=True, help="YYYY-MM-DD inclusive.")
@click.option("--tile", required=True, help="MGRS tile id (e.g. 42TUK). Stored in the parquet's 'tile' column.")
@click.option("--cloud-cover-max", default=80, show_default=True, type=int)
@click.option("--s2-max-scenes-per-month", default=4, show_default=True, type=int,
              help="Top-N cleanest S2 scenes per month. 0 disables cap.")
@click.option("--s1-max-scenes-per-month", default=6, show_default=True, type=int,
              help="First-N S1 scenes per month (chronological). 0 disables cap.")
@click.option("--out-dir", default=None, type=click.Path(),
              help="Defaults to outputs/finetune/extractions/<ref_id>/")
@click.option("--skip-s2", is_flag=True, default=False, help="Fill all S2 bands with NODATAVALUE.")
@click.option("--skip-s1", is_flag=True, default=False, help="Fill all S1 bands with NODATAVALUE.")
@click.option("--skip-dem", is_flag=True, default=False, help="Fill slope/elevation with NODATAVALUE.")
@click.option("--skip-agera5", is_flag=True, default=False, help="Fill AGERA5 cols with NODATAVALUE.")
@click.option("--max-samples", default=None, type=int, help="Cap sample count (smoketest).")
def extract(
    samples_path: str, ref_id: str, start_date: str, end_date: str,
    tile: str, cloud_cover_max: int,
    s2_max_scenes_per_month: int, s1_max_scenes_per_month: int,
    out_dir: str | None,
    skip_s2: bool, skip_s1: bool, skip_dem: bool, skip_agera5: bool,
    max_samples: int | None,
) -> None:
    """Extract one tile's worth of samples locally → CDSE-format geoparquet."""
    out_path_dir = Path(out_dir) if out_dir else (EXTRACTIONS_ROOT / ref_id)
    out_path_dir.mkdir(parents=True, exist_ok=True)

    samples = gpd.read_parquet(samples_path)
    if "geometry" not in samples.columns:
        raise click.ClickException("samples geoparquet must have a geometry column")
    if samples.crs is None or str(samples.crs) != "EPSG:4326":
        samples = samples.to_crs("EPSG:4326")
    if max_samples is not None and len(samples) > max_samples:
        samples = samples.iloc[:max_samples].reset_index(drop=True)
        _log.info("capped samples to %d (smoketest)", max_samples)
    _log.info("loaded %d samples from %s", len(samples), samples_path)

    bbox = _bbox_from_samples(samples)
    months = _months_between(pd.Timestamp(start_date), pd.Timestamp(end_date))
    _log.info("range: %s..%s -> %d months", start_date, end_date, len(months))

    s2_by_month = (
        {} if skip_s2
        else _build_s2_monthly(
            bbox, months,
            cloud_cover_max=cloud_cover_max,
            max_scenes_per_month=s2_max_scenes_per_month,
        )
    )
    s1_by_month = (
        {} if skip_s1
        else _build_s1_monthly(
            bbox, months,
            max_scenes_per_month=s1_max_scenes_per_month,
        )
    )
    dem_static = None if skip_dem else _build_dem_static(samples, bbox)
    agera5_by_month = (
        None if skip_agera5
        else _build_agera5_monthly(samples, months, start_date, end_date)
    )

    year = max(months).year
    df = _build_rows(
        samples, s2_by_month, s1_by_month, months,
        tile=tile, ref_id=ref_id,
        start_date=start_date, end_date=end_date, year=year,
        dem_static=dem_static,
        agera5_by_month=agera5_by_month,
    )

    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
    out_name = f"WORLDCEREAL_{ref_id}_{start_date}_{end_date}_{tile}_LOCAL.geoparquet"
    out_path = out_path_dir / out_name
    gdf.to_parquet(out_path, index=False)
    _log.info("wrote %s (%d rows)", out_path, len(gdf))


@cli.command("verify-schema")
@click.option("--reference", "ref_path", required=True, type=click.Path(exists=True))
@click.option("--candidate", "cand_path", required=True, type=click.Path(exists=True))
def verify_schema(ref_path: str, cand_path: str) -> None:
    """Compare a local-produced parquet against a CDSE-produced one."""
    ref = gpd.read_parquet(ref_path)
    cand = gpd.read_parquet(cand_path)

    ref_cols = set(ref.columns)
    cand_cols = set(cand.columns)
    missing = ref_cols - cand_cols
    extra = cand_cols - ref_cols
    print(f"reference columns: {len(ref_cols)}")
    print(f"candidate columns: {len(cand_cols)}")
    if missing:
        print(f"MISSING in candidate: {sorted(missing)}")
    if extra:
        print(f"EXTRA in candidate:  {sorted(extra)}")

    common = ref_cols & cand_cols
    mismatched = []
    for c in sorted(common):
        rd, cd = str(ref[c].dtype), str(cand[c].dtype)
        if rd != cd:
            mismatched.append((c, rd, cd))
    if mismatched:
        print("DTYPE MISMATCHES:")
        for c, rd, cd in mismatched:
            print(f"  {c}: ref={rd}  cand={cd}")
    else:
        print("dtypes match for all common columns.")


if __name__ == "__main__":
    cli()