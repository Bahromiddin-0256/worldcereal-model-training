"""Build a consensus-cleaned feature CSV for XGBoost retraining.

Filters outputs/xgboost_6win/features.csv to rows whose Mongo doc satisfies:
    crop_type == crop_type_xgboost == crop_type_tempcnn == crop_type_resnet50plus

This is the high-confidence subset where all three production models agree
with the truth label — used to retrain XGBoost on a noise-reduced sample.
"""

from __future__ import annotations

from pathlib import Path

import click
import numpy as np
import pandas as pd
from pymongo import MongoClient


CLASS_TO_INT = {"bugdoy": 0, "other": 1, "paxta": 2}


@click.command()
@click.option("--features",
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              default=Path("outputs/xgboost_6win/features.csv"),
              show_default=True)
@click.option("--ids",
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              default=Path("data/processed_regional_v6win_retrain/ids.npy"),
              show_default=True)
@click.option("--output",
              type=click.Path(dir_okay=False, path_type=Path),
              default=Path("outputs/xgboost_consensus/features.csv"),
              show_default=True)
@click.option("--uri", default="mongodb://localhost:27019", show_default=True)
@click.option("--db", default="gis-census", show_default=True)
@click.option("--collection", default="uzcosmos_flats", show_default=True)
def main(features: Path, ids: Path, output: Path,
         uri: str, db: str, collection: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)

    click.echo(f"Loading ids from {ids}")
    chip_ids = np.load(ids, allow_pickle=True)
    chip_ids = np.array([str(x) for x in chip_ids])
    click.echo(f"  {len(chip_ids):,} chip ids")

    click.echo(f"Querying {db}.{collection} for consensus rows")
    client = MongoClient(uri, serverSelectionTimeoutMS=10_000)
    col = client[db][collection]
    query = {
        "crop_type_xgboost": {"$exists": True},
        "$expr": {"$and": [
            {"$eq": ["$crop_type", "$crop_type_tempcnn"]},
            {"$eq": ["$crop_type", "$crop_type_resnet50plus"]},
            {"$eq": ["$crop_type", "$crop_type_xgboost"]},
        ]},
    }
    consensus = {}
    for d in col.find(query, {"_id": 1, "crop_type": 1}):
        consensus[str(d["_id"])] = d.get("crop_type")
    client.close()
    click.echo(f"  {len(consensus):,} consensus docs in Mongo")

    mask = np.array([cid in consensus for cid in chip_ids], dtype=bool)
    n_keep = int(mask.sum())
    click.echo(f"  {n_keep:,} of {len(chip_ids):,} chip ids match consensus set")

    click.echo(f"Loading features {features}")
    df = pd.read_csv(features)
    if len(df) != len(chip_ids):
        raise click.ClickException(
            f"row mismatch: features.csv has {len(df):,} rows but ids.npy "
            f"has {len(chip_ids):,}"
        )

    df_keep = df.iloc[mask].reset_index(drop=True)

    # Sanity: verify Mongo's truth label agrees with the integer label in features.csv
    kept_ids = chip_ids[mask]
    truth_int = np.array(
        [CLASS_TO_INT[consensus[i]] for i in kept_ids], dtype=np.int64
    )
    mismatched = int((truth_int != df_keep["label"].values).sum())
    if mismatched:
        click.echo(f"  WARNING: {mismatched:,} rows have label mismatch "
                   f"between features.csv and Mongo crop_type")
    else:
        click.echo("  label parity OK (features.csv label == Mongo crop_type)")

    df_keep.to_csv(output, index=False)
    click.echo(f"\nWrote {len(df_keep):,} rows × {df_keep.shape[1]-1} features → {output}")
    click.echo("Class distribution:")
    for k, v in df_keep["label"].value_counts().sort_index().items():
        name = {v: k for k, v in CLASS_TO_INT.items()}[int(k)]
        click.echo(f"  {int(k)} ({name}): {int(v):,}")


if __name__ == "__main__":
    main()
