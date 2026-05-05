"""Run ESA WorldCereal for the 2024-2025 season over Uzbekistan.

This is a thin wrapper around the ``worldcereal-classification`` package that
submits an openEO job on the Copernicus Data Space Ecosystem (CDSE). It
produces per-season GeoTIFFs that
``scripts/validate_uzcosmos_worldcereal.py`` then consumes.

Prerequisites:

1. Register for a CDSE account and note the openEO endpoint
   (https://dataspace.copernicus.eu). Processing credits are required — a
   country-scale run will consume a non-trivial chunk, so budget for it.
2. Install the worldcereal package into a dedicated venv (heavy deps)::

       python -m venv .venv-worldcereal
       source .venv-worldcereal/bin/activate
       pip install "worldcereal[notebooks] @ git+https://github.com/WorldCereal/worldcereal-classification.git"

3. The first openEO call will open a device-code login flow in your
   browser. Tokens are cached by the openEO client (``~/.config/openeo``).

Typical usage::

    python scripts/run_worldcereal.py cropland \\
        --bbox 55.998,37.184,73.148,45.590 \\
        --season 2024-10-01:2025-09-30 \\
        --out-dir data/worldcereal/2025

    python scripts/run_worldcereal.py croptype \\
        --bbox 55.998,37.184,73.148,45.590 \\
        --season 2024-10-01:2025-09-30 \\
        --wheat-window 2024-11-01:2025-04-30 \\
        --out-dir data/worldcereal/2025

For large AOIs, split by oblast — openEO batch jobs have per-job limits.
Pass a GeoJSON of oblast polygons via ``--aoi-file`` instead of ``--bbox``
to do one job per feature.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from wc_train.utils.logging import get_logger

_log = get_logger(__name__)


@dataclass(frozen=True)
class Window:
    start: str
    end: str

    @classmethod
    def parse(cls, raw: str) -> Window:
        if ":" not in raw:
            raise click.BadParameter(f"expected START:END, got {raw!r}")
        start, end = raw.split(":", 1)
        return cls(start.strip(), end.strip())


def _load_aoi(bbox: str | None, aoi_file: Path | None):
    """Return a GeoDataFrame with one row per AOI."""
    import geopandas as gpd
    from shapely.geometry import box

    if (bbox is None) == (aoi_file is None):
        raise click.UsageError("exactly one of --bbox or --aoi-file must be provided")

    if bbox is not None:
        lon_min, lat_min, lon_max, lat_max = (float(x) for x in bbox.split(","))
        gdf = gpd.GeoDataFrame(
            {"id": ["aoi"]},
            geometry=[box(lon_min, lat_min, lon_max, lat_max)],
            crs="EPSG:4326",
        )
        return gdf

    gdf = gpd.read_file(aoi_file)
    if "id" not in gdf.columns:
        gdf = gdf.assign(id=[f"aoi_{i}" for i in range(len(gdf))])
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)
    return gdf


def _import_worldcereal():
    """Load the worldcereal package lazily so --help works without it installed."""
    try:
        from openeo_gfmap import TemporalContext  # type: ignore[import-not-found]
        from worldcereal.job import WorldCerealTask  # type: ignore[import-not-found]
        from worldcereal.jobmanager import run_worldcereal_task  # type: ignore[import-not-found]
        from worldcereal.parameters import WorldCerealProductType  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - exercised manually
        raise click.ClickException(
            "worldcereal-classification is not installed. Install it with:\n"
            "  pip install 'worldcereal[notebooks] @ "
            "git+https://github.com/WorldCereal/worldcereal-classification.git'"
        ) from exc
    return TemporalContext, WorldCerealTask, run_worldcereal_task, WorldCerealProductType


@click.group()
def cli() -> None:
    """Submit WorldCereal classification jobs to CDSE openEO."""


@cli.command("cropland")
@click.option("--bbox", default=None, help="lon_min,lat_min,lon_max,lat_max (EPSG:4326)")
@click.option(
    "--aoi-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="GeoJSON of per-feature AOIs. Submits one job per feature.",
)
@click.option("--season", required=True, help="Full-year window, e.g. 2024-10-01:2025-09-30")
@click.option(
    "--out-dir",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Local output directory (GeoTIFFs land here once the openEO job finishes).",
)
def cropland_cmd(bbox: str | None, aoi_file: Path | None, season: str, out_dir: Path) -> None:
    """Generate a seasonal cropland / no-cropland raster."""
    window = Window.parse(season)
    temporal_context_cls, task_cls, run_task, product_type_cls = _import_worldcereal()

    aoi_gdf = _load_aoi(bbox, aoi_file)
    temporal = temporal_context_cls(window.start, window.end)
    out_dir.mkdir(parents=True, exist_ok=True)

    for _, row in aoi_gdf.iterrows():
        row_gdf = aoi_gdf.iloc[[row.name]]
        job_dir = out_dir / f"cropland_{row['id']}"
        job_dir.mkdir(exist_ok=True)
        _log.info("submitting WorldCereal cropland job for %s → %s", row["id"], job_dir)
        params = {
            "aoi_gdf": row_gdf,
            "output_dir": str(job_dir),
            "temporal_extent": temporal,
            "product_type": product_type_cls.CROPLAND,
        }
        run_task(task_cls.CLASSIFICATION, params)
        (job_dir / "params.json").write_text(
            json.dumps(
                {"season_start": window.start, "season_end": window.end, "aoi_id": row["id"]},
                indent=2,
            )
        )


@cli.command("croptype")
@click.option("--bbox", default=None, help="lon_min,lat_min,lon_max,lat_max (EPSG:4326)")
@click.option(
    "--aoi-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="GeoJSON of per-feature AOIs.",
)
@click.option("--season", required=True, help="Full-year window, e.g. 2024-10-01:2025-09-30")
@click.option(
    "--wheat-window",
    default=None,
    help="Winter-cereals season window, e.g. 2024-11-01:2025-04-30. "
    "Optional; omit to use the WorldCereal default for the AEZ.",
)
@click.option(
    "--out-dir",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
)
def croptype_cmd(
    bbox: str | None,
    aoi_file: Path | None,
    season: str,
    wheat_window: str | None,
    out_dir: Path,
) -> None:
    """Generate a crop-type raster (winter-cereals vs others)."""
    window = Window.parse(season)
    temporal_context_cls, task_cls, run_task, product_type_cls = _import_worldcereal()

    aoi_gdf = _load_aoi(bbox, aoi_file)
    temporal = temporal_context_cls(window.start, window.end)
    season_specs = None
    if wheat_window is not None:
        w = Window.parse(wheat_window)
        season_specs = {"winter-cereals": temporal_context_cls(w.start, w.end)}

    out_dir.mkdir(parents=True, exist_ok=True)

    for _, row in aoi_gdf.iterrows():
        row_gdf = aoi_gdf.iloc[[row.name]]
        job_dir = out_dir / f"croptype_{row['id']}"
        job_dir.mkdir(exist_ok=True)
        _log.info("submitting WorldCereal crop-type job for %s → %s", row["id"], job_dir)
        params: dict = {
            "aoi_gdf": row_gdf,
            "output_dir": str(job_dir),
            "temporal_extent": temporal,
            "product_type": product_type_cls.CROPTYPE,
        }
        if season_specs is not None:
            params["season_specifications"] = season_specs
        run_task(task_cls.CLASSIFICATION, params)
        (job_dir / "params.json").write_text(
            json.dumps(
                {
                    "season_start": window.start,
                    "season_end": window.end,
                    "wheat_window": wheat_window,
                    "aoi_id": row["id"],
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    cli()
