"""Filter uzcosmos polygons against WorldCereal cropland + wheat rasters.

Usage (from repo root)::

    .venv/bin/python scripts/validate_uzcosmos_worldcereal.py \\
        --cropland data/worldcereal/2025/uzbekistan_cropland.tif \\
        --wheat    data/worldcereal/2025/uzbekistan_wintercereals.tif \\
        --labels   data/labels/uzcosmos.geojson \\
        --out-dir  data/labels/worldcereal_2025

Writes three GeoJSON files (``accepted.geojson``, ``review.geojson``,
``rejected.geojson``) and a per-polygon summary CSV.

The thresholds mirror the defaults in
:class:`wc_train.data.worldcereal.Thresholds` but can be overridden via CLI.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from wc_train.data.worldcereal import (
    PolygonScore,
    Thresholds,
    Verdict,
    score_polygons,
)
from wc_train.utils.logging import get_logger

_log = get_logger(__name__)


def _load_features(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    feats = data.get("features") if isinstance(data, dict) else None
    if not feats:
        raise click.ClickException(f"{path} does not look like a GeoJSON FeatureCollection")
    return feats


def _parse_values(raw: str) -> tuple[int, ...]:
    return tuple(int(v.strip()) for v in raw.split(",") if v.strip())


def _write_split(
    features: list[dict],
    scores: list[PolygonScore],
    out_dir: Path,
) -> dict[Verdict, int]:
    buckets: dict[Verdict, list[dict]] = {v: [] for v in Verdict}
    for feat, score in zip(features, scores, strict=True):
        # Copy so the user's input file is not mutated.
        new_feat = dict(feat)
        new_props = dict(feat.get("properties") or {})
        new_props["worldcereal_verdict"] = score.verdict.value
        new_props["worldcereal_reason"] = score.reason
        new_props["worldcereal_cropland_fraction"] = score.cropland_fraction
        new_props["worldcereal_wheat_fraction"] = score.wheat_fraction
        new_props["worldcereal_pixel_count"] = score.n_pixels
        new_feat["properties"] = new_props
        buckets[score.verdict].append(new_feat)

    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[Verdict, int] = {}
    for verdict, feats in buckets.items():
        payload = {"type": "FeatureCollection", "features": feats}
        (out_dir / f"{verdict.value}.geojson").write_text(json.dumps(payload))
        counts[verdict] = len(feats)
    return counts


def _write_summary_csv(path: Path, scores: list[PolygonScore]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "polygon_id",
                "label",
                "label_normalised",
                "n_pixels",
                "cropland_fraction",
                "wheat_fraction",
                "verdict",
                "reason",
            ]
        )
        for s in scores:
            writer.writerow(
                [
                    s.polygon_id,
                    s.label,
                    s.label_normalised,
                    s.n_pixels,
                    s.cropland_fraction,
                    s.wheat_fraction,
                    s.verdict.value,
                    s.reason,
                ]
            )


@click.command()
@click.option(
    "--cropland",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="WorldCereal cropland GeoTIFF.",
)
@click.option(
    "--wheat",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="WorldCereal crop-type / winter-cereals GeoTIFF. Omit for cropland-only run.",
)
@click.option(
    "--labels",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Input GeoJSON FeatureCollection with polygons + crop_type properties.",
)
@click.option(
    "--out-dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Directory to write accepted/review/rejected GeoJSON + summary CSV.",
)
@click.option("--label-field", default="crop_type", show_default=True)
@click.option("--id-field", default="id", show_default=True)
@click.option(
    "--cropland-values",
    default="1",
    show_default=True,
    help="Comma-separated pixel values that mean 'cropland' in --cropland raster.",
)
@click.option(
    "--wheat-values",
    default="1",
    show_default=True,
    help="Comma-separated pixel values that mean 'wheat/winter-cereal' in --wheat raster.",
)
@click.option("--crop-accept", type=float, default=Thresholds.crop_accept, show_default=True)
@click.option("--crop-reject", type=float, default=Thresholds.crop_reject, show_default=True)
@click.option("--wheat-accept", type=float, default=Thresholds.wheat_accept, show_default=True)
@click.option("--wheat-reject", type=float, default=Thresholds.wheat_reject, show_default=True)
def main(
    cropland: Path,
    wheat: Path | None,
    labels: Path,
    out_dir: Path,
    label_field: str,
    id_field: str,
    cropland_values: str,
    wheat_values: str,
    crop_accept: float,
    crop_reject: float,
    wheat_accept: float,
    wheat_reject: float,
) -> None:
    thresholds = Thresholds(
        crop_accept=crop_accept,
        crop_reject=crop_reject,
        wheat_accept=wheat_accept,
        wheat_reject=wheat_reject,
    )
    features = _load_features(labels)
    _log.info("scoring %d polygons against %s", len(features), cropland)

    scores = score_polygons(
        features,
        cropland_raster=cropland,
        wheat_raster=wheat,
        label_field=label_field,
        id_field=id_field,
        cropland_positive=_parse_values(cropland_values),
        wheat_positive=_parse_values(wheat_values),
        thresholds=thresholds,
    )

    # score_polygons skips features without a label — realign for writing.
    labelled_features = [f for f in features if (f.get("properties") or {}).get(label_field) is not None]
    counts = _write_split(labelled_features, scores, out_dir)
    _write_summary_csv(out_dir / "summary.csv", scores)

    print(f"\nscored {len(scores):,} polygons from {labels}")
    for verdict in Verdict:
        print(f"  {verdict.value:<9}: {counts[verdict]:>7,}")

    reasons = Counter(s.reason for s in scores)
    print("\ntop reasons:")
    for reason, n in reasons.most_common(10):
        print(f"  {n:>7,}  {reason}")


if __name__ == "__main__":
    main()
