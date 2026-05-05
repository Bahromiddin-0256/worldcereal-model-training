"""Fine-tune Presto on uzcosmos polygons using WorldCereal point extractions.

Pipeline
--------
1. Read all per-tile GeoParquets in
   ``outputs/finetune/extractions/<ref_id>/`` (the output of
   ``scripts/extract_finetune_points.py submit``).
2. Map our 3 ewoc_codes to crop-type labels:
       1101010001 -> wheat        (bugdoy)
       1108000010 -> fibre_crops  (paxta)
       1100000000 -> other_crops  (other — cropland that's not wheat or cotton)
   1700000000 is also mapped to ``other_crops`` for backward-compatibility
   with parquets extracted before the ewoc_code was corrected.
3. Use WorldCereal's ``get_training_dfs_from_parquet`` to wide-pivot the
   monthly time series and split train/val/test by H3 cell so spatial
   leakage is controlled.
4. Two modes:

   * ``head``    — compute Presto embeddings (encoder frozen) + train a
     CatBoost head on top. Cheap, fast (~minutes on CPU).
   * ``finetune`` — unfreeze the encoder and run end-to-end training
     with WorldCereal's TorchTrainer + MLPHead. Heavier; needs a GPU
     for reasonable wall-clock.

Usage
-----
Once the smoketest extraction finishes, sanity-check head-only training::

    python scripts/finetune_presto.py head \\
        --ref-id 2025_uzcosmos_finetune_smoketest

Full fine-tune on the country-wide extraction::

    python scripts/finetune_presto.py finetune \\
        --ref-id 2025_uzcosmos_finetune_v1 \\
        --epochs 30 --batch-size 256 --lr 1e-4
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from wc_train.utils.logging import get_logger  # noqa: E402

_log = get_logger(__name__)


EXTRACTIONS_ROOT = ROOT / "outputs" / "finetune" / "extractions"
CHECKPOINTS_DIR = ROOT / "outputs" / "finetune" / "checkpoints"


# 3-class custom mapping for the uzcosmos fine-tune.
# "other_crops" is everything in uzcosmos that was checked as cropland but
# is neither wheat nor cotton — NOT non-cropland. The 1700000000 key is
# kept so parquets extracted before the ewoc_code was fixed still train
# without re-extraction.
UZCOSMOS_CLASS_MAPPING = {
    "UZCOSMOS3": {
        "1101010001": "wheat",        # bugdoy
        "1108000010": "fibre_crops",  # paxta
        "1100000000": "other_crops",  # other (correct ewoc_code, going forward)
        "1700000000": "other_crops",  # other (legacy — pre-fix extractions)
    }
}

DEFAULT_REF_ID = "2025_uzcosmos_finetune_v1"


def _list_parquets(ref_id: str) -> list[Path]:
    base = EXTRACTIONS_ROOT / ref_id
    if not base.exists():
        raise click.ClickException(f"no extraction folder at {base}")
    parquets = sorted(base.rglob("*.geoparquet"))
    if not parquets:
        raise click.ClickException(f"no .geoparquet files under {base}")
    return parquets


def _patch_worldcereal_check_faulty_samples() -> None:
    """Workaround for worldcereal 2.6.1 (commit 1103648) bug in
    ``utils.timeseries.process_parquet``: ``valid_time`` is strftime'd to
    str just before ``check_faulty_samples`` is called, but ``start_date``
    and ``end_date`` are only strftime'd *after* that call — so the
    comparison ``valid_time > end_date`` mixes str with datetime and pandas
    raises ``TypeError: Invalid comparison``.
    """
    from worldcereal.utils import timeseries as _ts

    if getattr(_ts.DataFrameValidator.check_faulty_samples, "_uzcosmos_patched", False):
        return

    _orig = _ts.DataFrameValidator.check_faulty_samples

    def _patched(df_wide, min_edge_buffer):
        for col in ("start_date", "end_date"):
            if col in df_wide.columns and pd.api.types.is_datetime64_any_dtype(df_wide[col]):
                df_wide = df_wide.assign(**{col: df_wide[col].dt.strftime("%Y-%m-%d")})
        out = _orig(df_wide, min_edge_buffer)
        for col in ("start_date", "end_date"):
            if col in out.columns and not pd.api.types.is_datetime64_any_dtype(out[col]):
                out = out.assign(**{col: pd.to_datetime(out[col])})
        return out

    _patched._uzcosmos_patched = True
    _ts.DataFrameValidator.check_faulty_samples = staticmethod(_patched)


def _load_splits(ref_id: str, debug: bool = False):
    """Return (train_df, val_df, test_df) ready for embedding extraction."""
    _patch_worldcereal_check_faulty_samples()
    from worldcereal.train.data import get_training_dfs_from_parquet

    parquets = _list_parquets(ref_id)
    _log.info("loading %d parquet files for ref_id=%s", len(parquets), ref_id)
    train_df, val_df, test_df = get_training_dfs_from_parquet(
        parquet_files=parquets,
        timestep_freq="month",
        max_timesteps_trim="auto",
        use_valid_time=True,
        finetune_classes="UZCOSMOS3",
        class_mappings=UZCOSMOS_CLASS_MAPPING,
        debug=debug,
    )
    _log.info(
        "splits: train=%d val=%d test=%d", len(train_df), len(val_df), len(test_df)
    )
    return train_df, val_df, test_df


@click.group()
def cli() -> None:
    """Fine-tune Presto on uzcosmos extractions."""


@cli.command("head")
@click.option("--ref-id", default=DEFAULT_REF_ID, show_default=True)
@click.option("--repeats", default=3, show_default=True, type=int)
@click.option(
    "--head",
    type=click.Choice(["catboost", "mlp"]),
    default="catboost",
    show_default=True,
    help="Type of head to train on top of frozen Presto embeddings.",
)
@click.option("--debug", is_flag=True, default=False)
def head_only(ref_id: str, repeats: int, head: str, debug: bool) -> None:
    """Train a head on FROZEN Presto embeddings (cheap baseline)."""
    from worldcereal.train.data import compute_embeddings_from_splits

    train_df, val_df, test_df = _load_splits(ref_id, debug=debug)

    _log.info("computing Presto embeddings for all splits...")
    emb_train = compute_embeddings_from_splits(
        train_df, val_df, test_df, repeats=repeats
    )
    # compute_embeddings_from_splits returns the concatenation; tag splits
    # via a new column it already adds (typically "split").
    out_dir = CHECKPOINTS_DIR / ref_id / f"head_{head}"
    out_dir.mkdir(parents=True, exist_ok=True)
    emb_path = out_dir / "embeddings.parquet"
    emb_train.to_parquet(emb_path)
    _log.info("wrote embeddings -> %s", emb_path)

    feature_cols = [c for c in emb_train.columns if c.startswith("presto_ft_")]
    assert feature_cols, "no embedding columns found (col prefix 'presto_ft_')"
    label_col = "finetune_class"

    # ``.to_numpy()`` with explicit dtype unwraps any pyarrow-backed
    # extension arrays — catboost rejects ArrowStringArray.
    X = emb_train[feature_cols].to_numpy(dtype="float32")
    y = emb_train[label_col].astype(str).to_numpy()
    splits = emb_train["split"].astype(str).to_numpy()  # 'train' | 'val' | 'test'

    train_mask = splits == "train"
    val_mask = splits == "val"
    test_mask = splits == "test"

    if head == "catboost":
        from catboost import CatBoostClassifier

        clf = CatBoostClassifier(
            iterations=2000,
            learning_rate=0.05,
            depth=6,
            loss_function="MultiClass",
            eval_metric="TotalF1",
            random_seed=42,
            verbose=100,
            early_stopping_rounds=50,
        )
        clf.fit(
            X[train_mask],
            y[train_mask],
            eval_set=(X[val_mask], y[val_mask]) if val_mask.any() else None,
        )
        clf.save_model(str(out_dir / "catboost.cbm"))
    else:
        from sklearn.metrics import classification_report
        from sklearn.neural_network import MLPClassifier

        clf = MLPClassifier(
            hidden_layer_sizes=(256, 128),
            max_iter=200,
            random_state=42,
            verbose=True,
        )
        clf.fit(X[train_mask], y[train_mask])

    # Evaluate on test
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
    )

    y_pred = clf.predict(X[test_mask])
    if y_pred.ndim == 2:  # CatBoost MultiClass returns column vector
        y_pred = y_pred.ravel()
    acc = accuracy_score(y[test_mask], y_pred)
    report = classification_report(y[test_mask], y_pred, digits=3, zero_division=0)
    cm = confusion_matrix(y[test_mask], y_pred)

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "ref_id": ref_id,
                "head": head,
                "repeats": repeats,
                "test_accuracy": float(acc),
                "test_n": int(test_mask.sum()),
                "report": report,
                "confusion_matrix": cm.tolist(),
                "labels": sorted(set(y[test_mask])),
            },
            indent=2,
        )
    )
    click.echo(f"\nTest accuracy: {acc:.4f}  (n={test_mask.sum()})")
    click.echo(f"\n{report}")
    click.echo(f"\nmetrics + model -> {out_dir}")


@cli.command("finetune")
@click.option("--ref-id", default=DEFAULT_REF_ID, show_default=True)
@click.option("--epochs", default=30, show_default=True, type=int)
@click.option("--batch-size", default=256, show_default=True, type=int)
@click.option("--lr", default=2e-5, show_default=True, type=float)
@click.option("--num-workers", default=4, show_default=True, type=int)
@click.option("--epochs", "epochs", default=30, show_default=True, type=int)
@click.option("--patience", default=10, show_default=True, type=int)
@click.option(
    "--unfreeze-epoch",
    default=5,
    show_default=True,
    type=int,
    help="Epoch at which the Presto encoder is unfrozen (head-only before this).",
)
@click.option("--debug", is_flag=True, default=False)
def finetune(
    ref_id: str,
    epochs: int,
    batch_size: int,
    lr: float,
    num_workers: int,
    patience: int,
    unfreeze_epoch: int,
    debug: bool,
) -> None:
    """End-to-end fine-tune: Presto encoder + MLP head via worldcereal run_finetuning.

    Staged unfreeze: encoder is frozen for the first `unfreeze_epoch` epochs
    (head-only warm-up), then unfrozen for the remainder.  Trains to
    val-loss early stopping with patience `patience`.
    """
    import torch
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import ExponentialLR
    from torch.utils.data import DataLoader
    from worldcereal.train.data import build_presto_backbone
    from worldcereal.train.datasets import WorldCerealLabelledDataset
    from worldcereal.train.downstream import MLPHead
    from worldcereal.train.finetuning_utils import collate_fn, run_finetuning
    from prometheo.finetune import Hyperparams

    train_df, val_df, test_df = _load_splits(ref_id, debug=debug)

    classes_list = sorted(train_df["finetune_class"].astype(str).unique().tolist())
    n_classes = len(classes_list)
    _log.info("n_classes=%d classes=%s", n_classes, classes_list)

    # Derive num_timesteps from the pivot output rather than hardcoding.
    # available_timesteps was added by process_parquet; fall back to 16 if absent.
    num_timesteps = int(
        train_df["available_timesteps"].max()
        if "available_timesteps" in train_df.columns
        else 16
    )
    _log.info("num_timesteps=%d", num_timesteps)

    def _make_ds(df, augment: bool) -> WorldCerealLabelledDataset:
        return WorldCerealLabelledDataset(
            dataframe=df,
            num_timesteps=num_timesteps,
            timestep_freq="month",
            task_type="multiclass",
            num_outputs=1,
            classes_list=classes_list,
            augment=augment,
        )

    train_ds = _make_ds(train_df, augment=True)
    val_ds = _make_ds(val_df, augment=False)
    test_ds = _make_ds(test_df, augment=False)

    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_fn, drop_last=True,
    )
    val_dl = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn,
    )
    test_dl = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn,
    )

    backbone = build_presto_backbone()

    class PrestoClassifier(torch.nn.Module):
        """Presto encoder + MLP head as a single nn.Module.

        Naming matches what worldcereal's run_finetuning expects when saving
        encoder-only checkpoints (looks for model.backbone first).
        """
        def __init__(self, backbone, head):
            super().__init__()
            self.backbone = backbone
            self.head = head

        def forward(self, x):
            embeddings = self.backbone(x)
            return self.head(embeddings)

    head = MLPHead(
        in_dim=backbone.encoder.embedding_size,
        num_classes=n_classes,
        hidden_dim=256,
        dropout=0.2,
    )
    model = PrestoClassifier(backbone, head)

    from prometheo.utils import device as _device
    model = model.to(_device)
    _log.info("model moved to device=%s", _device)

    out_dir = CHECKPOINTS_DIR / ref_id / "finetune"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Guard against run_finetuning refusing to overwrite an existing checkpoint.
    stale = out_dir / "presto_uzcosmos3.pt"
    if stale.exists():
        stale.unlink()
        _log.warning("removed stale checkpoint %s before re-run", stale)

    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = ExponentialLR(optimizer, gamma=0.99)

    hyperparams = Hyperparams(
        lr=lr,
        max_epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        num_workers=num_workers,
    )

    _log.info(
        "starting run_finetuning: epochs=%d lr=%g unfreeze_epoch=%d patience=%d",
        epochs, lr, unfreeze_epoch, patience,
    )
    model = run_finetuning(
        model=model,
        train_dl=train_dl,
        val_dl=val_dl,
        experiment_name="presto_uzcosmos3",
        output_dir=out_dir,
        optimizer=optimizer,
        scheduler=scheduler,
        hyperparams=hyperparams,
        loss_fn=torch.nn.CrossEntropyLoss(),
        freeze_layers=["backbone.encoder"],
        unfreeze_epoch=unfreeze_epoch,
    )

    # ── Evaluate on held-out test split ──────────────────────────────────────
    from worldcereal.train.finetuning_utils import _unpack_predictor_batch
    import numpy as np
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for batch in test_dl:
            predictors, _ = _unpack_predictor_batch(batch)
            logits = model(predictors)                         # [B, n_classes]
            targets = predictors.label.to(_device)
            targets = targets.long().squeeze()                 # squeeze all singletons → [B]
            mask = targets != 65535
            pred_cls = logits[mask].argmax(dim=-1).cpu().numpy()
            true_cls = targets[mask].cpu().numpy()
            all_preds.append(pred_cls)
            all_true.append(true_cls)

    y_pred = np.concatenate(all_preds).ravel().astype(int)
    y_true = np.concatenate(all_true).ravel().astype(int)

    cls_arr = np.array(classes_list)
    y_pred_names = cls_arr[y_pred].tolist()
    y_true_names = cls_arr[y_true].tolist()

    acc = accuracy_score(y_true_names, y_pred_names)
    report = classification_report(y_true_names, y_pred_names, digits=3, zero_division=0)
    cm = confusion_matrix(y_true_names, y_pred_names, labels=classes_list)

    metrics = {
        "ref_id": ref_id,
        "test_accuracy": float(acc),
        "test_n": int(len(y_true)),
        "report": report,
        "confusion_matrix": cm.tolist(),
        "labels": classes_list,
        "hyperparams": {
            "lr": lr, "epochs": epochs, "batch_size": batch_size,
            "patience": patience, "unfreeze_epoch": unfreeze_epoch,
        },
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))

    click.echo(f"\nTest accuracy: {acc:.4f}  (n={len(y_true)})")
    click.echo(f"\n{report}")
    click.echo(f"\ndone -> {out_dir}")


@cli.command("predict")
@click.option(
    "--ref-id",
    default=DEFAULT_REF_ID,
    show_default=True,
    help="ref_id of a CDSE extraction to score (must already be on disk).",
)
@click.option(
    "--model-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Folder containing catboost.cbm + metrics.json (from `head` command).",
)
@click.option(
    "--out",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Output parquet with predictions (default: <model-dir>/predictions.parquet).",
)
def predict(ref_id: str, model_dir: Path, out: Path | None) -> None:
    """Score a CDSE extraction with the fine-tuned CatBoost head.

    Workflow for inference on new polygons:

      1. Build a GeoJSON of target polygons (any source).
      2. Run ``extract_finetune_points.py submit`` to extract their
         Presto inputs from CDSE — produces parquets in the same layout
         as the training data.
      3. Run this command to score those parquets and emit per-polygon
         class predictions.
    """
    from catboost import CatBoostClassifier
    from worldcereal.train.data import (
        WorldCerealTrainingDataset,
        build_presto_backbone,
        dataset_to_embeddings,
        get_training_dfs_from_parquet,
    )

    # Load model
    cbm = model_dir / "catboost.cbm"
    if not cbm.exists():
        raise click.ClickException(f"missing {cbm}")
    clf = CatBoostClassifier()
    clf.load_model(str(cbm))

    _patch_worldcereal_check_faulty_samples()

    # Build a single dataset from all parquets — we don't care about the
    # train/val/test split here, we want to score everything.
    parquets = _list_parquets(ref_id)
    train_df, val_df, test_df = get_training_dfs_from_parquet(
        parquet_files=parquets,
        timestep_freq="month",
        finetune_classes="UZCOSMOS3",
        class_mappings=UZCOSMOS_CLASS_MAPPING,
    )
    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    ds = WorldCerealTrainingDataset(
        dataframe=full_df,
        num_timesteps=12,
        timestep_freq="month",
        task_type="multiclass",
        augment=False,
    )

    backbone = build_presto_backbone()
    emb_df = dataset_to_embeddings(ds, presto_model=backbone, batch_size=2048)

    feature_cols = [c for c in emb_df.columns if c.startswith("presto_ft_")]
    X = emb_df[feature_cols].values
    pred = clf.predict(X).ravel()
    proba = clf.predict_proba(X)
    classes = list(clf.classes_)

    result = pd.DataFrame(
        {
            "sample_id": emb_df["sample_id"].values,
            "lat": emb_df["lat"].values,
            "lon": emb_df["lon"].values,
            "true_class": emb_df["finetune_class"].values
            if "finetune_class" in emb_df.columns
            else None,
            "pred_class": pred,
        }
    )
    for i, c in enumerate(classes):
        result[f"prob_{c}"] = proba[:, i]

    out_path = out or (model_dir / "predictions.parquet")
    result.to_parquet(out_path, index=False)

    click.echo(f"\nwrote {len(result):,} predictions -> {out_path}")
    click.echo("\nclass distribution (predicted):")
    for cls, n in result["pred_class"].value_counts().items():
        click.echo(f"  {cls:<15} {n:>6,}")
    if result["true_class"].notna().any():
        from sklearn.metrics import accuracy_score, classification_report

        m = result["true_class"].notna()
        acc = accuracy_score(result.loc[m, "true_class"], result.loc[m, "pred_class"])
        click.echo(f"\nOverall accuracy on labelled samples: {acc:.4f} (n={m.sum():,})")
        click.echo(
            classification_report(
                result.loc[m, "true_class"],
                result.loc[m, "pred_class"],
                digits=3,
                zero_division=0,
            )
        )


@cli.command("inspect")
@click.option("--ref-id", default=DEFAULT_REF_ID, show_default=True)
def inspect(ref_id: str) -> None:
    """List parquets + show a row count + per-class breakdown."""
    parquets = _list_parquets(ref_id)
    click.echo(f"{len(parquets)} parquet files under {EXTRACTIONS_ROOT/ref_id}")
    total = 0
    samples = 0
    by_code: dict[int, int] = {}
    for p in parquets:
        df = pd.read_parquet(p, columns=["sample_id", "ewoc_code"])
        total += len(df)
        samples += df["sample_id"].nunique()
        for code, n in df.groupby("ewoc_code")["sample_id"].nunique().items():
            by_code[int(code)] = by_code.get(int(code), 0) + int(n)
        click.echo(
            f"  {p.relative_to(EXTRACTIONS_ROOT/ref_id)}  "
            f"rows={len(df):,} samples={df['sample_id'].nunique():,}"
        )
    click.echo(f"\nTOTAL rows={total:,} unique_samples={samples:,}")
    click.echo("ewoc_code -> sample count:")
    for code, n in sorted(by_code.items()):
        click.echo(f"  {code} -> {n:,}")


if __name__ == "__main__":
    cli()
