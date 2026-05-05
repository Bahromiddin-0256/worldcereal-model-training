# worldcereal-model-training

WorldCereal label-agreement filtering and Presto fine-tuning for crop classification.

Split out of `ai-train-on-gis-data` — see that repo for the broader Sentinel-2 / uzcosmos training pipeline.

## Layout

- `src/wc_train/data/worldcereal.py` — zonal-stats validator that scores polygons against ESA WorldCereal rasters.
- `scripts/run_worldcereal.py` — produces the WorldCereal cropland/crop-type rasters via CDSE openEO.
- `scripts/validate_uzcosmos_worldcereal.py` — runs the validator against a polygon GeoJSON.
- `scripts/download_worldcereal_finetune.py` — downloads WorldCereal finetune samples.
- `scripts/extract_finetune_points.py` — extracts point samples for Presto finetuning.
- `scripts/finetune_presto.py` — fine-tunes Presto on extracted samples.
- `scripts/score_chips_presto.py` — scores chips with a fine-tuned Presto model.
- `scripts/build_consensus_features.py` — builds consensus features from Mongo.
- `models/worldcereal/` — pretrained / fine-tuned model artifacts.

## Install

```bash
pip install -e .
```
