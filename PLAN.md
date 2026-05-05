# WorldCereal Presto Fine-Tuning Plan

Resuming the run that paused 2026-05-01. Source of truth for what's done, what's
next, and the exact commands.

## Where we are

**Goal:** fine-tune Presto on uzcosmos polygons → 3-class crop classifier
(`wheat` / `fibre_crops` / `no_crop`) → score Mongo + chip cubes country-wide.

**Class mapping** (`scripts/finetune_presto.py:UZCOSMOS_CLASS_MAPPING`):

| ewoc_code | label | uzcosmos source |
|---|---|---|
| 1101010001 | wheat | bugdoy |
| 1108000010 | fibre_crops | paxta |
| 1700000000 | no_crop | other |

**Tuman selection (frozen):** 13 tumans, one per viloyat,
~84,377 polygons total — `outputs/finetune/picks.json`.

**Smoketest result** (Qo'rg'ontepa, Andijan, 300 test samples, CatBoost head on
frozen Presto embeddings):

```
test_accuracy: 0.933
              precision   recall  f1
 fibre_crops    0.932     0.960   0.946
     no_crop    0.888     0.950   0.918
       wheat    0.989     0.890   0.937
```

Main confusion: **9 wheat → no_crop** (recall 0.89 on wheat). Worth keeping an eye
on once the country-wide head trains.

**CDSE extraction state:**

| ref_id | parquets | status |
|---|---|---|
| `2025_uzcosmos_finetune_smoketest` | — | merged for smoketest |
| `2025_uzcosmos_finetune_v1` | — | superseded by v2 |
| `2025_uzcosmos_finetune_v2_w0` | **29 / 31** | ✅ done, 2 jobs errored |
| `2025_uzcosmos_finetune_v2_w1` | **8 / 31** | ⛔ orchestration died mid-run |

`worldcereal_merged_extractions.parquet/` contains only `ref_id=v2_w0`.

---

## Step 1 — Resume wave-1 extraction

CDSE jobs from May 1 are expired; re-submit. The submit script is idempotent on
already-finished tiles (it consults `job_tracking.csv`), so resuming should pick
up only the 21 not-started + 2 mid-flight jobs.

```bash
cd /home/prog/PycharmProjects/worldcereal-model-training
nohup python scripts/extract_finetune_points.py submit \
    --ref-id 2025_uzcosmos_finetune_v2_w1 \
    --samples outputs/finetune/samples_worldcereal_w1.geoparquet \
    > logs/finetune_cdse_w1_resume.log 2>&1 &
```

**Acceptance:** `job_tracking.csv` shows status=finished for all 31 jobs in
`extractions/2025_uzcosmos_finetune_v2_w1/`. Tail the log; expect ~2-4 hours
walltime depending on CDSE queue.

**Decide on the 2 errored w0 jobs:** check
`extractions/2025_uzcosmos_finetune_v2_w0/failed_jobs/` — if the failing tiles
are sparsely populated, accept the loss; otherwise re-submit them via a
single-job re-run.

---

## Step 2 — Re-merge extractions

After w1 finishes, regenerate the partitioned parquet so it covers both waves.

```bash
python scripts/extract_finetune_points.py merge \
    --ref-ids 2025_uzcosmos_finetune_v2_w0,2025_uzcosmos_finetune_v2_w1 \
    --out outputs/finetune/extractions/worldcereal_merged_extractions.parquet
```

(If the `merge` subcommand doesn't exist on `extract_finetune_points.py`, the
WorldCereal `get_training_dfs_from_parquet` accepts a directory of parquets
directly — `finetune_presto.py` already does this per-`ref_id`. The merge step is
only needed if we want a single-file artifact for downstream tooling.)

**Acceptance:** parquet has both `ref_id=v2_w0` and `ref_id=v2_w1` partitions;
total row count ≈ sum of per-tile parquets.

---

## Step 3 — Head-only baseline on full dataset

Cheap (~minutes on CPU). Validates the data pipeline end-to-end and gives a
baseline to beat with full fine-tuning.

```bash
python scripts/finetune_presto.py head \
    --ref-id 2025_uzcosmos_finetune_v2_w0 \
    --head catboost \
    --repeats 3
```

Then once w1 is merged, run on the combined ref:

```bash
python scripts/finetune_presto.py head \
    --ref-id 2025_uzcosmos_finetune_v2_combined \
    --head catboost \
    --repeats 3
```

**Acceptance gates** (before moving to full fine-tune):

- macro-F1 ≥ 0.90 across 13 tumans (smoketest hit 0.93 on one tuman; expect a
  drop with regional diversity).
- per-class recall ≥ 0.85 for all three classes.
- H3-cell-based train/val/test split — confirm no leakage by checking unique
  H3 cells per split in the log.

If the head model misses these gates, **do not** proceed to full fine-tuning —
debug data quality first (likely culprits: WorldCereal label-agreement filter
too lenient, tumans with skewed class balance, mis-labelled polygons).

---

## Step 4 — Full Presto fine-tune

GPU required. Encoder unfrozen, MLPHead, WorldCereal `TorchTrainer`.

```bash
python scripts/finetune_presto.py finetune \
    --ref-id 2025_uzcosmos_finetune_v2_combined \
    --epochs 30 \
    --batch-size 256 \
    --lr 1e-4 \
    --num-workers 4
```

**Hyperparameter notes:**

- `lr 1e-4` is the WorldCereal default for Presto fine-tuning. If the head
  baseline already saturates, try `lr 5e-5` to avoid catastrophic forgetting of
  the pretrained encoder.
- `epochs 30` with early stopping on val macro-F1 — 30 is a ceiling, not a
  target.
- `batch-size 256` is fine for a single 24GB GPU; reduce to 128 if OOM.
- Save the encoder + head separately so we can swap heads without retraining.

**Acceptance gates:**

- Test macro-F1 ≥ head baseline + 0.02 (otherwise full fine-tune isn't earning
  its compute).
- Wheat recall ≥ 0.92 (the smoketest's weak spot).
- Val curves stable — no sign of catastrophic forgetting (sudden drop after
  early epochs).

Keep the **best** checkpoint by val macro-F1, not the final.

---

## Step 5 — Score uzcosmos chips & validate

Apply the fine-tuned model to held-out tumans and compare against the existing
in-Mongo XGBoost predictions (per memory: XGBoost is currently best at 0.973
acc / 0.973 macro-F1 on 74k Andijan rows).

```bash
python scripts/score_chips_presto.py \
    --model-dir outputs/finetune/checkpoints/2025_uzcosmos_finetune_v2_combined/finetune \
    --chips-dir <path-to-sentinel-cube> \
    --out outputs/finetune/predictions/v2_combined.parquet
```

**Comparison plan:** join Presto predictions with Mongo XGBoost predictions on
`sample_id`; compute confusion + agreement rate. Budget: don't claim Presto is
"better" unless macro-F1 beats XGBoost by ≥ 0.01 on a held-out non-Andijan tuman
— per the project memory, label noise sits at ~80–85% accuracy, so any
small delta is in the noise floor.

---

## Step 6 — Productionise

Only after Step 5 results are clean:

1. **Export model:** zip `outputs/finetune/checkpoints/<run>/finetune/` →
   `models/worldcereal/<run>.zip` (mirroring the existing
   `phase_ii_multitask.zip` structure).
2. **Mongo update:** field `presto_uzcosmos_v2_pred` (parallel to existing
   `xgb_pred`, `resnet50plus_pred`, etc.); never overwrite XGBoost predictions.
3. **Document** in this repo's README which checkpoint is current and which is
   superseded.

---

## Risks & open questions

- **Label noise ceiling.** Per project memory, uzcosmos labels are ~80-85%
  accurate. Smoketest at 93% on a single tuman is already at/above the
  label-quality ceiling; country-wide we may *appear* to plateau or regress
  because the labels themselves disagree. Don't push hyperparameters past
  diminishing returns.
- **Train/test leakage by tuman.** H3-cell split helps but tumans are clustered
  geographically. Consider a leave-one-tuman-out eval as a robustness check
  before claiming generalisation.
- **CDSE quota.** Wave-1 resume might queue overnight; check
  `running_per_backend` in the log to see if we're rate-limited.
- **Errored w0 jobs.** Two tiles failed in w0; if those are non-empty (sample
  count > 100), they materially shift the dataset. Inspect
  `failed_jobs/` and decide.
- **Class imbalance.** `picks.json` shows skew toward `bugdoy` in some tumans
  (Sirdaryo: 962 wheat / 460 paxta) and toward `other` in Qoraqalpogiston
  (11,476 / 15,266 = 75%). The CatBoost head handles this fine but the
  end-to-end fine-tune may need class weighting.
