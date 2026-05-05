"""Submit CDSE openEO POINT_WORLDCEREAL extractions for Presto fine-tuning.

Pipeline:

1. Read ``outputs/finetune/finetune_polygons.geojson`` (one tuman per
   viloyat; balanced ~500 polygons / class / tuman).
2. Map uzcosmos crop labels onto WorldCereal ewoc_codes:
       bugdoy  -> 1101010001  (unspecified_winter_wheat)
       paxta   -> 1108000010  (cotton)
       other   -> 1100000000  (generic temporary_crops — cropland that is
                                not wheat or cotton in the uzcosmos campaign)
3. Convert polygons to centroids and add the columns the WorldCereal
   ``run_extractions`` API expects (sample_id, ewoc_code, valid_time, …).
4. Write the GeoParquet that the API consumes.
5. Optionally call ``worldcereal.extract.common.run_extractions`` which
   submits one openEO batch job per S2 tile to CDSE. Each job extracts
   the full Presto input stack — 12 monthly S2 + S1 sigma0 + AgERA5
   temp/precip + Copernicus DEM + lat/lon — at the centroid of every
   polygon in the tile, and downloads a per-tile GeoParquet.

Outputs land in ``outputs/finetune/extractions/<ref_id>/``.

Auth: the first run opens a device-code OIDC flow in the browser; tokens
are cached at ``~/.config/openeo``.

Usage
-----
Sanity check (build samples df, no submission)::

    python scripts/extract_finetune_points.py prepare

Run a small pilot (one tuman, e.g. Andijan)::

    python scripts/extract_finetune_points.py submit \\
        --filter-tuman-code 1703220 \\
        --max-locations-per-job 100

Full run (all 13 tumans)::

    python scripts/extract_finetune_points.py submit
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import click
import geopandas as gpd
import pandas as pd
from shapely.geometry import shape

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from wc_train.utils.logging import get_logger  # noqa: E402

_log = get_logger(__name__)


# uzcosmos crop_type -> WorldCereal ewoc_code.
# - bugdoy is winter wheat in Uzbekistan; harvested ~June.
# - paxta is cotton; harvested ~Sep-Oct.
# - "other" inside a uzcosmos campaign means the polygon was checked,
#   is cropland, but is neither wheat nor cotton. Map to the generic
#   temporary_crops code so WorldCereal treats it as cropland (1100000000
#   maps to "temporary_crops" in the croptype mappings, whereas 1700000000
#   would be non_cropland and gets dropped as "ignore").
EWOC_CODES = {
    "bugdoy": 1101010001,
    "paxta": 1108000010,
    "other": 1100000000,
}

# Uzbekistan 2024/2025 growing season midpoint. WorldCereal grabs
# valid_time ± 275 days of monthly composites — 2025-06-01 covers
# Sep-2024 through Feb-2026 which spans winter wheat AND cotton seasons.
DEFAULT_VALID_TIME = "2025-06-01"

INPUT_GEOJSON = ROOT / "outputs" / "finetune" / "finetune_polygons.geojson"
OUT_DIR = ROOT / "outputs" / "finetune"
SAMPLES_PARQUET = OUT_DIR / "samples_worldcereal.geoparquet"


def _load_polygons(geojson_path: Path = INPUT_GEOJSON) -> gpd.GeoDataFrame:
    if not geojson_path.exists():
        raise click.ClickException(
            f"missing {geojson_path} — run the export step first."
        )
    raw = json.loads(geojson_path.read_text())
    feats = raw["features"]
    rows = []
    for f in feats:
        p = f.get("properties") or {}
        rows.append(
            {
                "_id": p.get("_id") or f.get("id"),
                "tuman_code": p.get("tuman_code"),
                "tuman": p.get("tuman"),
                "viloyat": p.get("viloyat"),
                "crop_type": p.get("crop_type"),
                "geometry": shape(f["geometry"]),
            }
        )
    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    return gdf


def build_samples_df(
    valid_time: str = DEFAULT_VALID_TIME,
    filter_tuman_code: int | None = None,
    input_geojson: Path = INPUT_GEOJSON,
) -> gpd.GeoDataFrame:
    gdf = _load_polygons(input_geojson)
    if filter_tuman_code is not None:
        gdf = gdf[gdf["tuman_code"] == filter_tuman_code].copy()
        _log.info("filtered to tuman_code=%s -> %d polygons", filter_tuman_code, len(gdf))

    if gdf.empty:
        raise click.ClickException("no polygons after filtering")

    unknown = sorted(set(gdf["crop_type"]) - set(EWOC_CODES))
    if unknown:
        raise click.ClickException(f"unmapped crop_type values: {unknown}")

    # Centroid in equal-area projection to avoid lat-distortion bias,
    # then back to WGS84 for the openEO call.
    centroids = gdf.to_crs("EPSG:6933").geometry.centroid.to_crs("EPSG:4326")

    out = gpd.GeoDataFrame(
        {
            "sample_id": gdf["_id"].astype(str).values,
            "ewoc_code": gdf["crop_type"].map(EWOC_CODES).astype("int64").values,
            "valid_time": pd.to_datetime(valid_time).date(),
            # sampling_ewoc_code: the broader category we sampled from.
            # Re-using ewoc_code is fine for our 3-class setup.
            "sampling_ewoc_code": gdf["crop_type"].map(EWOC_CODES).astype("int64").values,
            "h3_l3_cell": "",  # not used by the point extractor
            "irrigation_status": 0,  # unknown
            "quality_score_lc": 100,  # treat all as max-quality LC labels
            "quality_score_ct": 100,
            "extract": 1,  # >=1 means "include"
            "tuman_code": gdf["tuman_code"].values,
            "viloyat": gdf["viloyat"].values,
            "crop_type": gdf["crop_type"].values,
            "geometry": centroids.values,
        },
        crs="EPSG:4326",
    )
    return out


@click.group()
def cli() -> None:
    """Build samples + submit CDSE point extractions for fine-tune."""


@cli.command("prepare")
@click.option("--valid-time", default=DEFAULT_VALID_TIME, show_default=True)
@click.option(
    "--filter-tuman-code",
    type=int,
    default=None,
    help="If set, restrict samples to a single tuman (smoke test).",
)
@click.option(
    "--input-geojson",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=INPUT_GEOJSON,
    show_default=True,
    help="Path to the polygon GeoJSON (built by the Mongo export step).",
)
def prepare(valid_time: str, filter_tuman_code: int | None, input_geojson: Path) -> None:
    """Build the samples GeoParquet that run_extractions consumes."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = build_samples_df(valid_time=valid_time, filter_tuman_code=filter_tuman_code, input_geojson=input_geojson)
    out_path = SAMPLES_PARQUET
    if filter_tuman_code is not None:
        out_path = OUT_DIR / f"samples_worldcereal_tc{filter_tuman_code}.geoparquet"
    df.to_parquet(out_path)

    click.echo(f"wrote {len(df):,} samples -> {out_path}")
    click.echo("\nclass distribution:")
    for cls, n in df["crop_type"].value_counts().items():
        click.echo(f"  {cls:<10} {n:>6,}")
    click.echo("\nviloyat distribution:")
    for v, n in df["viloyat"].value_counts().items():
        click.echo(f"  {v:<25} {n:>6,}")


@cli.command("submit")
@click.option("--valid-time", default=DEFAULT_VALID_TIME, show_default=True)
@click.option(
    "--filter-tuman-code",
    type=int,
    default=None,
    help="If set, restrict samples to a single tuman.",
)
@click.option(
    "--ref-id",
    default="2025_uzcosmos_finetune_v1",
    show_default=True,
    help=(
        "Identifier carried into the output folder + STAC metadata. "
        "MUST start with a 4-digit year — WorldCereal's post-job action "
        "does ``int(ref_id.split('_')[0])`` and crashes otherwise."
    ),
)
@click.option(
    "--max-locations-per-job",
    type=int,
    default=500,
    show_default=True,
    help="Points per CDSE batch job (smaller = more jobs, less risk of OOM).",
)
@click.option(
    "--parallel-jobs",
    type=int,
    default=2,
    show_default=True,
    help="Concurrent CDSE batch jobs; raise carefully — eats credits faster.",
)
@click.option(
    "--restart-failed",
    is_flag=True,
    default=False,
    help="Restart errored / start_failed jobs from a previous run.",
)
@click.option(
    "--input-geojson",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=INPUT_GEOJSON,
    show_default=True,
    help="Path to the polygon GeoJSON (built by the Mongo export step).",
)
@click.option(
    "--backend",
    type=click.Choice(["cdse", "terrascope"], case_sensitive=False),
    default="cdse",
    show_default=True,
    help="openEO backend to submit jobs to.",
)
@click.option(
    "--num-workers",
    type=int,
    default=1,
    show_default=True,
    help="Total number of parallel worker processes splitting the sample set.",
)
@click.option(
    "--worker-id",
    type=int,
    default=0,
    show_default=True,
    help="0-based worker index (0..num-workers-1). Output dir gets _w<id> suffix when num-workers>1.",
)
def submit(
    valid_time: str,
    filter_tuman_code: int | None,
    ref_id: str,
    max_locations_per_job: int,
    parallel_jobs: int,
    restart_failed: bool,
    input_geojson: Path,
    backend: str,
    num_workers: int,
    worker_id: int,
) -> None:
    """Submit one point-extraction batch job per S2 tile."""
    if num_workers < 1 or not (0 <= worker_id < num_workers):
        raise click.ClickException(
            f"worker_id must be in [0, {num_workers}); got worker_id={worker_id} num_workers={num_workers}"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = build_samples_df(valid_time=valid_time, filter_tuman_code=filter_tuman_code, input_geojson=input_geojson)

    # Stable worker split using the int value of the Mongo ObjectId sample_id
    # so the same sample always lands in the same shard across processes
    # (Python's hash() is randomized per-process and would shuffle splits).
    if num_workers > 1:
        def _shard(sid: str) -> int:
            try:
                return int(sid, 16) % num_workers
            except ValueError:
                return int(hashlib.md5(sid.encode()).hexdigest(), 16) % num_workers
        shard_mask = df["sample_id"].astype(str).map(lambda s: _shard(s) == worker_id)
        df = df[shard_mask].reset_index(drop=True)
        ref_id = f"{ref_id}_w{worker_id}"
        click.echo(
            f"shard worker_id={worker_id}/{num_workers} -> {len(df):,} samples, ref_id={ref_id}"
        )

    suffix = f"_w{worker_id}" if num_workers > 1 else ""
    if filter_tuman_code is not None:
        out_path = OUT_DIR / f"samples_worldcereal_tc{filter_tuman_code}{suffix}.geoparquet"
    else:
        out_path = OUT_DIR / f"samples_worldcereal{suffix}.geoparquet"
    df.to_parquet(out_path)
    click.echo(f"prepared {len(df):,} samples -> {out_path}")

    # Heavy imports happen here so that `prepare` works without auth/network.
    from openeo_gfmap.backend import Backend
    from worldcereal.extract.common import run_extractions
    from worldcereal.stac.constants import ExtractionCollection

    backend_enum = Backend[backend.upper()]

    output_folder = OUT_DIR / "extractions" / ref_id
    output_folder.mkdir(parents=True, exist_ok=True)

    click.echo(f"submitting POINT_WORLDCEREAL extractions to {backend_enum.name} -> {output_folder}")
    click.echo(
        f"  max_locations_per_job={max_locations_per_job} parallel_jobs={parallel_jobs}"
    )

    run_extractions(
        collection=ExtractionCollection.POINT_WORLDCEREAL,
        output_folder=output_folder,
        samples_df_path=out_path,
        ref_id=ref_id,
        max_locations_per_job=max_locations_per_job,
        parallel_jobs=parallel_jobs,
        restart_failed=restart_failed,
        backend=backend_enum,
    )

    click.echo(f"DONE -> {output_folder}")


if __name__ == "__main__":
    cli()
