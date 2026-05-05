"""Score processed_regional_mt chips with the WorldCereal-pretrained Presto encoder.

Pipeline:

    1. Load the 45-channel x 64 x 64 chips and reshape each chip into
       Presto's required ``(T=12 timesteps, B=17 bands)`` layout:

       * Spatial dimensions are mean-pooled (standard for polygon-level Presto
         inference).
       * The 9 Sentinel-2 reflectance bands we have (B02-B12, excluding B8A)
         land in the S2 slots of Presto's band vector; SAR, AgERA5, SRTM, and
         B8A slots are zero-filled **and masked** so the encoder ignores them.
         NDVI is recomputed internally by Presto from B4/B8 — we do not supply
         the EVI/NDWI/NDRE/MSI/NBR channels in the user's 45-channel stack.
       * The 3 time windows (Apr-May / Jun-Jul / Aug-Sep from the Uzbekistan
         S2 config) are placed at Presto months April, June, August; all
         other months are masked.

    2. Run Presto's encoder in batches, producing a 128-d embedding per chip.
    3. Train an XGBoost classifier with K-fold cross-validation on the
       embeddings + declared labels; collect out-of-fold probabilities.
    4. Emit per-sample verdicts in the same schema as
       ``scripts/detect_label_noise.py`` (accepted / review / rejected).

**Why this is a different signal than** ``detect_label_noise.py``:
the feature space here comes from a self-supervised global pre-training run
(~1M polygons, 7 continents); it cannot have overfit to uzcosmos label noise.
Samples rejected by both scripts are very high-confidence mislabels.

Caveat: we lack S1/weather/SRTM and the winter half of the growing season,
so the encoder operates on a degraded input compared to the CDSE pipeline.
Treat probabilities as ordering, not as calibrated WorldCereal outputs.

Usage::

    # One-time: download encoder weights (3 MB)
    curl -L -o data/pretrained/presto_default_model.pt \\
        https://huggingface.co/nasaharvest/presto/resolve/main/default_model.pt

    .venv-worldcereal/bin/python scripts/score_chips_presto.py \\
        --data-dir data/processed_regional_mt \\
        --model-path data/pretrained/presto_default_model.pt \\
        --out-csv  data/processed_regional_mt/presto_verdicts.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from wc_train.utils.logging import get_logger

_log = get_logger(__name__)


# Matches configs/data/uzbekistan_s2.yaml — 9 S2 bands then 6 indices,
# repeated for each of the 3 time windows.
USER_BANDS: tuple[str, ...] = ("B02", "B03", "B04", "B05", "B06", "B07", "B08", "B11", "B12")
USER_INDICES: tuple[str, ...] = ("NDVI", "EVI", "NDWI", "NDRE", "MSI", "NBR")
N_WINDOWS = 3
CHANNELS_PER_WINDOW = len(USER_BANDS) + len(USER_INDICES)  # 15

# Uzbekistan S2 extraction window is Apr-Sep — pin the 3 chips to April, June,
# August of Presto's 12-month calendar. Month indices are 0-based (Jan=0).
WINDOW_TIMESTEPS: tuple[int, ...] = (0, 2, 4)
START_MONTH = 3  # April


def _build_band_mapping() -> tuple[list[int], list[int], list[float]]:
    """Return (src_idx_per_band, mask_per_group, BAND_EXPANSION).

    src_idx_per_band[i] is the column in the user's 15-channel window that
    maps to Presto ``BANDS[i]`` (18 entries, pre-B9 removal). -1 means
    "no data; will be zero-filled and masked".
    """
    # Presto ordering of BANDS (18 entries, before B9 removal).
    presto_bands = (
        "VV", "VH",
        "B2", "B3", "B4",
        "B5", "B6", "B7",
        "B8",
        "B8A",
        "B9",
        "B11", "B12",
        "temperature_2m", "total_precipitation",
        "elevation", "slope",
        "NDVI",
    )
    # User's band → user channel index.
    user_idx = {b: i for i, b in enumerate(USER_BANDS)}
    # User names use "B02" / "B11" etc.; Presto uses "B2" / "B11".
    alias = {
        "B2": "B02", "B3": "B03", "B4": "B04", "B5": "B05",
        "B6": "B06", "B7": "B07", "B8": "B08",
        "B11": "B11", "B12": "B12",
    }
    src = [user_idx.get(alias.get(b, b), -1) for b in presto_bands]
    # NDVI slot gets recomputed by Presto, so a zero there is fine.
    return src, [], []


def _reshape_to_presto(images: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert ``(N, 45, H, W)`` chips to ``(N, 12, 18)`` eo_data + ``(N, 12, 9)`` group-mask.

    Values are the raw-DN S2 reflectance x 10000 that Presto's normaliser
    expects. Channels we don't have are zero-filled. The returned mask is
    per-band-group (9 groups); expand with ``BAND_EXPANSION`` downstream.
    """
    n, c, _, _ = images.shape
    if c != N_WINDOWS * CHANNELS_PER_WINDOW:
        raise ValueError(f"expected {N_WINDOWS * CHANNELS_PER_WINDOW} channels, got {c}")

    # Spatial pool: mean over HxW → (N, C).
    pooled = images.reshape(n, c, -1).mean(axis=2)

    # Unstack into (N, n_windows, channels_per_window).
    pooled = pooled.reshape(n, N_WINDOWS, CHANNELS_PER_WINDOW)

    src_idx, _, _ = _build_band_mapping()
    eo = np.zeros((n, 12, len(src_idx)), dtype=np.float32)

    for w_i, t in enumerate(WINDOW_TIMESTEPS):
        window = pooled[:, w_i, :]  # (N, 15)
        for band_out, band_src in enumerate(src_idx):
            if band_src >= 0:
                eo[:, t, band_out] = window[:, band_src]

    # Per-group mask: 1 = missing. Group order matches Presto's
    # BANDS_GROUPS_IDX: S1, S2_RGB, S2_Red_Edge, S2_NIR_10m, S2_NIR_20m,
    # S2_SWIR, ERA5, SRTM, NDVI.
    # We have S2_RGB (B2,B3,B4), S2_Red_Edge (B5,B6,B7), S2_NIR_10m (B8),
    # S2_SWIR (B11,B12), and NDVI (recomputed). We lack the rest.
    group_mask_present = np.array(
        [1, 0, 0, 0, 1, 0, 1, 1, 0], dtype=np.float32
    )  # 1=missing, 0=present
    mask = np.ones((n, 12, 9), dtype=np.float32)
    mask[:] = 1.0  # everything masked
    for t in WINDOW_TIMESTEPS:
        mask[:, t, :] = group_mask_present

    return eo, mask


def _encode(
    model,
    eo: np.ndarray,
    mask_grp: np.ndarray,
    latlons: np.ndarray,
    start_month: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Run Presto encoder → 128-d embeddings. ``mask_grp`` is per-group (9 groups)."""
    from presto.dataops import BAND_EXPANSION, S1_S2_ERA5_SRTM, DynamicWorld2020_2021

    # Presto's normalise takes (N, T, 18) in BANDS order → drops B9 → (N, T, 17).
    eo_norm = S1_S2_ERA5_SRTM.normalize(eo)

    # Expand mask from 9 groups to 17 bands.
    mask_bands = np.repeat(mask_grp, BAND_EXPANSION, axis=-1)

    n = eo_norm.shape[0]
    dynamic_world = np.full((n, 12), DynamicWorld2020_2021.class_amount, dtype=np.int64)
    months = np.full((n,), start_month, dtype=np.int64)

    model.eval()
    out = np.empty((n, model.encoder.embedding_size), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            s = slice(i, i + batch_size)
            x = torch.from_numpy(eo_norm[s]).float().to(device)
            dw = torch.from_numpy(dynamic_world[s]).long().to(device)
            ll = torch.from_numpy(latlons[s]).float().to(device)
            m = torch.from_numpy(mask_bands[s]).float().to(device)
            mo = torch.from_numpy(months[s]).long().to(device)
            emb = model.encoder(x, dynamic_world=dw, mask=m, latlons=ll, month=mo)
            out[s] = emb.cpu().numpy()
            if (i // batch_size) % 10 == 0:
                _log.info("  encoded %d / %d", min(i + batch_size, n), n)
    return out


def _oof_probs(
    features: np.ndarray,
    labels: np.ndarray,
    n_classes: int,
    n_splits: int,
    seed: int,
) -> np.ndarray:
    import xgboost as xgb
    from sklearn.model_selection import StratifiedKFold

    oof = np.zeros((len(labels), n_classes), dtype=np.float32)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    params = {
        "objective": "multi:softprob",
        "num_class": n_classes,
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "max_depth": 6,
        "eta": 0.08,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": seed,
        "verbosity": 0,
    }
    for fold, (tr, va) in enumerate(skf.split(features, labels), 1):
        dtrain = xgb.DMatrix(features[tr], label=labels[tr])
        dval = xgb.DMatrix(features[va], label=labels[va])
        classes, counts = np.unique(labels[tr], return_counts=True)
        cw = {int(c): len(tr) / (n_classes * k) for c, k in zip(classes, counts, strict=True)}
        dtrain.set_weight(np.array([cw[int(y)] for y in labels[tr]], dtype=np.float32))
        booster = xgb.train(
            params,
            dtrain,
            num_boost_round=500,
            evals=[(dval, "val")],
            early_stopping_rounds=30,
            verbose_eval=False,
        )
        pred = booster.predict(dval)
        if pred.ndim == 1:
            pred = pred.reshape(-1, n_classes)
        oof[va] = pred
        _log.info("fold %d/%d  val_mlogloss=%.4f", fold, n_splits, booster.best_score)
    return oof


def _verdicts(labels, oof, accept_prob, reject_prob):
    predicted = oof.argmax(axis=1)
    p_declared = oof[np.arange(len(labels)), labels]
    p_predicted = oof[np.arange(len(labels)), predicted]

    verdict = np.full(len(labels), "review", dtype=object)
    reason = np.full(len(labels), "low_confidence_or_near_boundary", dtype=object)

    accept = (predicted == labels) & (p_declared >= accept_prob)
    reject = (predicted != labels) & (p_predicted >= reject_prob)
    verdict[accept] = "accepted"
    reason[accept] = "presto_embedding_agrees"
    verdict[reject] = "rejected"
    reason[reject] = "presto_embedding_confidently_disagrees"

    return pd.DataFrame(
        {
            "sample_idx": np.arange(len(labels)),
            "declared_label": labels,
            "predicted_label": predicted,
            "p_declared": p_declared.round(4),
            "p_predicted": p_predicted.round(4),
            "margin": (p_declared - p_predicted).round(4),
            "verdict": verdict,
            "reason": reason,
        }
    )


@click.command()
@click.option("--data-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option("--model-path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--out-csv", type=click.Path(dir_okay=False, path_type=Path), default=None)
@click.option("--embeddings-path", type=click.Path(dir_okay=False, path_type=Path), default=None,
              help="Where to cache the 128-d embeddings; skip encoding on re-runs if present.")
@click.option("--center-lat", type=float, default=41.5, show_default=True,
              help="Latitude to feed Presto's position encoder (Uzbekistan centre).")
@click.option("--center-lon", type=float, default=64.5, show_default=True)
@click.option("--batch-size", type=int, default=2048, show_default=True)
@click.option("--device", type=click.Choice(["auto", "cpu", "cuda"]), default="auto", show_default=True)
@click.option("--n-splits", type=int, default=5, show_default=True)
@click.option("--accept-prob", type=float, default=0.65, show_default=True)
@click.option("--reject-prob", type=float, default=0.70, show_default=True)
@click.option("--seed", type=int, default=42, show_default=True)
@click.option("--limit", type=int, default=0, show_default=True, help="0 = full dataset.")
def main(
    data_dir: Path,
    model_path: Path,
    out_csv: Path | None,
    embeddings_path: Path | None,
    center_lat: float,
    center_lon: float,
    batch_size: int,
    device: str,
    n_splits: int,
    accept_prob: float,
    reject_prob: float,
    seed: int,
    limit: int,
) -> None:
    from presto.presto import Presto

    images = np.load(data_dir / "images.npy", mmap_mode="r")
    labels = np.load(data_dir / "labels.npy").astype(np.int64)
    if limit:
        images = images[:limit]
        labels = labels[:limit]
    n_samples = len(labels)
    _log.info("images: %s  labels: %s", images.shape, labels.shape)

    embeddings_path = embeddings_path or (data_dir / "presto_embeddings.npy")
    if embeddings_path.exists() and not limit:
        _log.info("loading cached embeddings from %s", embeddings_path)
        embeddings = np.load(embeddings_path)
        if len(embeddings) != n_samples:
            raise click.ClickException(
                f"cached embeddings have {len(embeddings)} rows, dataset has {n_samples}. "
                f"Delete {embeddings_path} to re-encode."
            )
    else:
        _log.info("reshaping chips for Presto input layout")
        eo, mask_grp = _reshape_to_presto(np.asarray(images))
        latlons = np.tile(np.array([center_lat, center_lon], dtype=np.float32), (n_samples, 1))

        if device == "auto":
            torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            torch_device = torch.device(device)
        _log.info("loading Presto encoder from %s  (device=%s)", model_path, torch_device)
        model = Presto.load_pretrained(model_path, strict=True).to(torch_device)

        _log.info("encoding %d chips (batch=%d)", n_samples, batch_size)
        embeddings = _encode(model, eo, mask_grp, latlons, START_MONTH, batch_size, torch_device)
        if not limit:
            embeddings_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(embeddings_path, embeddings)
            _log.info("cached embeddings → %s", embeddings_path)

    _log.info("training k-fold XGBoost on %d-dim embeddings", embeddings.shape[1])
    n_classes = int(labels.max()) + 1
    oof = _oof_probs(embeddings, labels, n_classes=n_classes, n_splits=n_splits, seed=seed)

    df = _verdicts(labels, oof, accept_prob=accept_prob, reject_prob=reject_prob)
    out_csv = out_csv or (data_dir / "presto_verdicts.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    print(f"\nwrote {out_csv}")
    counts = df["verdict"].value_counts().to_dict()
    print(f"  total    : {n_samples:>7,}")
    for v in ("accepted", "review", "rejected"):
        print(f"  {v:<9}: {counts.get(v, 0):>7,}")

    print("\ndeclared → verdict counts:")
    print(pd.crosstab(df["declared_label"], df["verdict"]).to_string())
    print(f"\noverall OOF accuracy: {(oof.argmax(axis=1) == labels).mean():.4f}")


if __name__ == "__main__":
    main()
