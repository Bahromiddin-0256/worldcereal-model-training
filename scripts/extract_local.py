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
- v1: S2-L2A monthly median composites + S1-RTC monthly mean (Planetary
  Computer STAC). Already-calibrated S1-RTC sidesteps SAR processing.
- DEM (slope/elevation) and AgERA5 (precip/tmean) columns are emitted with
  ``NODATAVALUE = 65535`` — ``process_parquet`` tolerates them.

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
) -> dict[pd.Timestamp, dict[str, xr.DataArray]]:
    """All COG reads are clipped to ``bbox`` (WGS84) — see ``_open_clipped``."""
    """Returns {month_ts: {band_name: 2D DataArray}} — uint16 DN scale.

    Stacks all scenes overlapping each month, applies SCL cloud mask, takes
    median per band. Skips a month if there are zero clear pixels.
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
        _log.info("S2 month %s: %d scenes", m.date(), len(items))

        scene_stacks: list[xr.DataArray] = []
        for it in items:
            try:
                scl = _open_clipped(_signed_href(it, "SCL"), bbox).squeeze("band")
                cloud = scl.isin(list(SCL_DROP))
                bands = []
                for b in S2_BAND_NAMES:
                    da = _open_clipped(_signed_href(it, b), bbox).squeeze("band")
                    da = da.rio.reproject_match(scl).where(~cloud, NODATAVALUE)
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

def _build_s1_monthly(
    bbox: tuple[float, float, float, float],
    months: list[pd.Timestamp],
) -> dict[pd.Timestamp, dict[str, xr.DataArray]]:
    """S1-RTC is already terrain-corrected sigma0 (linear). Monthly mean →
    uint16 sigma0-encoded.
    """
    items_all = _stac_search(
        collection="sentinel-1-rtc",
        bbox=bbox,
        start=str(months[0].date()),
        end=str((months[-1] + pd.offsets.MonthEnd(0)).date()),
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
        _log.info("S1 month %s: %d scenes", m.date(), len(items))

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

        for i in range(n_samples):
            row: dict = {
                "feature_index": fi,
                "sample_id": sample_ids[i],
                "timestamp": m,
                **{c: int(s2_vals[c][i]) for c in S2_PARQUET_COLS},
                **{c: int(s1_vals[c][i]) for c in S1_PARQUET_COLS},
                "slope": NODATAVALUE,
                "elevation": NODATAVALUE,
                "AGERA5-PRECIP": NODATAVALUE,
                "AGERA5-TMEAN": NODATAVALUE,
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
@click.option("--out-dir", default=None, type=click.Path(),
              help="Defaults to outputs/finetune/extractions/<ref_id>/")
def extract(
    samples_path: str, ref_id: str, start_date: str, end_date: str,
    tile: str, cloud_cover_max: int, out_dir: str | None,
) -> None:
    """Extract one tile's worth of samples locally → CDSE-format geoparquet."""
    out_path_dir = Path(out_dir) if out_dir else (EXTRACTIONS_ROOT / ref_id)
    out_path_dir.mkdir(parents=True, exist_ok=True)

    samples = gpd.read_parquet(samples_path)
    if "geometry" not in samples.columns:
        raise click.ClickException("samples geoparquet must have a geometry column")
    if samples.crs is None or str(samples.crs) != "EPSG:4326":
        samples = samples.to_crs("EPSG:4326")
    _log.info("loaded %d samples from %s", len(samples), samples_path)

    bbox = _bbox_from_samples(samples)
    months = _months_between(pd.Timestamp(start_date), pd.Timestamp(end_date))
    _log.info("range: %s..%s -> %d months", start_date, end_date, len(months))

    s2_by_month = _build_s2_monthly(bbox, months, cloud_cover_max=cloud_cover_max)
    s1_by_month = _build_s1_monthly(bbox, months)

    year = max(months).year
    df = _build_rows(
        samples, s2_by_month, s1_by_month, months,
        tile=tile, ref_id=ref_id,
        start_date=start_date, end_date=end_date, year=year,
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