"""Stratified-by-tuman sampling from gis-census.uzcosmos_flats.

For each ``(tuman_code, crop_type)`` group, random-sample N polygons
(default 17) and emit a GeoJSON in the same shape that
``extract_finetune_points.py prepare`` expects:

    properties = {_id, tuman_code, tuman, viloyat, crop_type}
    geometry   = parsed from geom_2 (WKT)

Why stratify by tuman instead of by viloyat?
- The current v2 set covers 13 tumans (1 per viloyat) and the model
  may have learned region-specific quirks. A small sample drawn from
  every tuman in Uz spreads training across ~180 districts → expected
  to generalise better at country-scale inference time.

Usage::

    python scripts/sample_polygons_by_tuman.py sample \\
        --per-class-per-tuman 17 \\
        --output outputs/finetune/finetune_polygons_v3.geojson

    python scripts/sample_polygons_by_tuman.py validate \\
        --geojson outputs/finetune/finetune_polygons_v3.geojson
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import click

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from wc_train.utils.logging import get_logger  # noqa: E402

_log = get_logger(__name__)


DEFAULT_MONGO_URL = os.getenv("MONGODB_URL") or "mongodb://localhost:27019"
DEFAULT_DB = "gis-census"
DEFAULT_COLL = "uzcosmos_flats"
TARGET_CROP_TYPES = ("bugdoy", "paxta", "other")


def _connect(url: str):
    from pymongo import MongoClient  # noqa: PLC0415
    client = MongoClient(url, serverSelectionTimeoutMS=5000)
    client.admin.command("ping")
    return client


def _wkt_to_geojson(wkt: str) -> dict | None:
    import shapely.wkt as _wkt  # noqa: PLC0415
    import shapely.geometry as _geom  # noqa: PLC0415
    try:
        return _geom.mapping(_wkt.loads(wkt))
    except Exception:
        return None


@click.group()
def cli() -> None:
    """Stratified sampling from uzcosmos_flats."""


@cli.command("sample")
@click.option("--mongo-url", default=DEFAULT_MONGO_URL, show_default=True)
@click.option("--db", default=DEFAULT_DB, show_default=True)
@click.option("--collection", default=DEFAULT_COLL, show_default=True)
@click.option("--per-class-per-tuman", default=17, show_default=True, type=int)
@click.option(
    "--output",
    default=str(ROOT / "outputs" / "finetune" / "finetune_polygons_v3.geojson"),
    show_default=True,
    type=click.Path(),
)
@click.option(
    "--min-per-group",
    default=5,
    show_default=True,
    type=int,
    help="Drop groups with fewer than this many polygons available.",
)
def sample(
    mongo_url: str, db: str, collection: str,
    per_class_per_tuman: int, output: str, min_per_group: int,
) -> None:
    """Pick a stratified subset and write GeoJSON."""
    _log.info("connecting to %s", mongo_url)
    client = _connect(mongo_url)
    coll = client[db][collection]
    _log.info("collection has ~%d docs", coll.estimated_document_count())

    # Per-group counts so we can warn / drop tiny groups.
    counts = list(coll.aggregate(
        [
            {"$match": {
                "crop_type": {"$in": list(TARGET_CROP_TYPES)},
                "geom_2": {"$exists": True, "$nin": [None, ""]},
            }},
            {"$group": {
                "_id": {"t": "$tuman_code", "c": "$crop_type"},
                "n": {"$sum": 1},
            }},
        ],
        allowDiskUse=True,
    ))
    _log.info("found %d (tuman x crop_type) groups", len(counts))

    keep: list[tuple] = []
    dropped = 0
    short = 0
    for r in counts:
        n = r["n"]
        if n < min_per_group:
            dropped += 1
            continue
        if n < per_class_per_tuman:
            short += 1
        keep.append((r["_id"]["t"], r["_id"]["c"], min(n, per_class_per_tuman)))
    _log.info(
        "groups: keep=%d  dropped<%d=%d  shorter_than_target=%d",
        len(keep), min_per_group, dropped, short,
    )

    # Per-group $sample. Many small queries — cheap when tuman_code is indexed.
    features: list[dict] = []
    skipped_geom = 0
    for tuman_code, crop_type, target_n in keep:
        docs = coll.aggregate(
            [
                {"$match": {"tuman_code": tuman_code, "crop_type": crop_type,
                            "geom_2": {"$exists": True, "$nin": [None, ""]}}},
                {"$sample": {"size": target_n}},
                {"$project": {
                    "_id": 1, "geom_2": 1, "crop_type": 1,
                    "tuman_code": 1, "tuman": 1, "viloyat": 1,
                }},
            ],
        )
        for d in docs:
            geom = _wkt_to_geojson(d.get("geom_2", ""))
            if not geom:
                skipped_geom += 1
                continue
            features.append({
                "type": "Feature",
                "geometry": geom,
                "id": str(d["_id"]),
                "properties": {
                    "_id": str(d["_id"]),
                    "tuman_code": d.get("tuman_code"),
                    "tuman": d.get("tuman"),
                    "viloyat": d.get("viloyat"),
                    "crop_type": d.get("crop_type"),
                },
            })

    _log.info("collected %d features (skipped %d unparseable WKTs)",
              len(features), skipped_geom)

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "type": "FeatureCollection",
        "features": features,
    }))
    _log.info("wrote %s", out_path)

    # Summary printout.
    by_class = Counter(f["properties"]["crop_type"] for f in features)
    by_viloyat = Counter(f["properties"].get("viloyat") for f in features)
    by_tuman = Counter(f["properties"]["tuman_code"] for f in features)
    click.echo(f"\ntotal features: {len(features):,}")
    click.echo("by crop_type:")
    for c, n in by_class.most_common():
        click.echo(f"  {c:<8} {n:>6,}")
    click.echo(f"unique tumans: {len(by_tuman)}")
    click.echo(f"unique viloyats: {len(by_viloyat)}")


@cli.command("validate")
@click.option("--geojson", required=True, type=click.Path(exists=True))
def validate(geojson: str) -> None:
    """Print per-tuman / per-class coverage stats."""
    raw = json.loads(Path(geojson).read_text())
    feats = raw["features"]
    per_tuman: dict = defaultdict(Counter)
    per_viloyat: dict = defaultdict(Counter)
    for f in feats:
        p = f["properties"]
        per_tuman[p["tuman_code"]][p["crop_type"]] += 1
        per_viloyat[p.get("viloyat")][p["crop_type"]] += 1

    click.echo(f"features: {len(feats):,}")
    click.echo(f"tumans:   {len(per_tuman)}")
    click.echo(f"viloyats: {len(per_viloyat)}")
    click.echo()

    # Tumans missing one or more target classes.
    incomplete = [t for t, cnt in per_tuman.items()
                  if any(c not in cnt for c in TARGET_CROP_TYPES)]
    click.echo(f"tumans missing >=1 target class: {len(incomplete)}")
    if incomplete[:5]:
        click.echo("  e.g. " + ", ".join(map(str, incomplete[:5])))
    click.echo()

    click.echo("per viloyat × crop_type:")
    for v in sorted(per_viloyat, key=lambda x: -sum(per_viloyat[x].values())):
        cnts = per_viloyat[v]
        click.echo(f"  {v!s:<35}  bugdoy={cnts.get('bugdoy',0):>4}  "
                   f"paxta={cnts.get('paxta',0):>4}  other={cnts.get('other',0):>4}")


if __name__ == "__main__":
    cli()
