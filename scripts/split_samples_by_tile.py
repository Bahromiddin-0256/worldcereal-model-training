"""Split a samples geoparquet by S2 MGRS tile.

For each sample point, computes the MGRS 100km tile id (e.g. ``42TUK``)
and writes one parquet per tile so ``extract_local.py`` can be invoked
in parallel — each tile call only loads its own subset rather than
treating every sample's bbox as the AOI.

Usage::

    python scripts/split_samples_by_tile.py \\
        --samples outputs/finetune/samples_worldcereal_v3.geoparquet \\
        --out-dir outputs/finetune/per_tile_v3
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import click
import geopandas as gpd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from wc_train.utils.logging import get_logger  # noqa: E402

_log = get_logger(__name__)


def _mgrs_tile_for_point(lat: float, lon: float, mgrs_obj) -> str:
    """Return the 5-char MGRS tile id (e.g. '42TUK') for one lon/lat."""
    full = mgrs_obj.toMGRS(lat, lon, MGRSPrecision=0)
    return full[:5]


@click.command()
@click.option(
    "--samples", "samples_path",
    required=True, type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "--out-dir",
    required=True, type=click.Path(file_okay=False),
)
@click.option(
    "--manifest",
    default=None, type=click.Path(),
    help="Optional JSON written next to per-tile parquets summarising "
         "tile -> sample count.",
)
def main(samples_path: str, out_dir: str, manifest: str | None) -> None:
    import mgrs as _mgrs  # noqa: PLC0415
    m = _mgrs.MGRS()

    gdf = gpd.read_parquet(samples_path)
    if gdf.crs is None or str(gdf.crs) != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")
    _log.info("loaded %d samples from %s", len(gdf), samples_path)

    lats = gdf.geometry.y.values
    lons = gdf.geometry.x.values
    tiles = [_mgrs_tile_for_point(la, lo, m) for la, lo in zip(lats, lons)]
    gdf = gdf.assign(tile=tiles)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tile_counts: dict[str, int] = {}
    for tile, sub in gdf.groupby("tile", sort=True):
        sub = sub.drop(columns=["tile"]).reset_index(drop=True)
        tile_path = out / f"samples_v3_{tile}.geoparquet"
        sub.to_parquet(tile_path)
        tile_counts[tile] = len(sub)
        _log.info("  %s -> %4d samples (%s)", tile, len(sub), tile_path.name)

    summary = {
        "total_samples": len(gdf),
        "tile_count": len(tile_counts),
        "samples_per_tile": dict(sorted(tile_counts.items(),
                                        key=lambda kv: -kv[1])),
    }
    manifest_path = Path(manifest) if manifest else (out / "manifest.json")
    manifest_path.write_text(json.dumps(summary, indent=2))
    _log.info("wrote manifest %s", manifest_path)

    click.echo(f"\ntotal: {len(gdf):,} samples in {len(tile_counts)} tiles")
    click.echo("largest tiles:")
    for tile, n in list(summary["samples_per_tile"].items())[:10]:
        click.echo(f"  {tile}: {n:>4}")


if __name__ == "__main__":
    main()
