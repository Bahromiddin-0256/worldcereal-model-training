"""Run ``extract_local.py extract`` once per S2 tile in parallel.

Reads ``manifest.json`` produced by ``split_samples_by_tile.py``, then
spawns up to ``--parallel`` concurrent ``extract_local.py extract``
subprocesses. Each tile gets:

- its own per-tile samples parquet (already split)
- a per-tile log file under ``logs/``
- skip-if-output-exists idempotency for resume after failure

Usage::

    python scripts/run_local_extractions.py \\
        --tile-dir outputs/finetune/per_tile_v3 \\
        --ref-id 2025_uzcosmos_finetune_v3 \\
        --start-date 2024-08-01 --end-date 2026-03-31 \\
        --min-samples-per-tile 30 \\
        --parallel 4
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import click

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from wc_train.utils.logging import get_logger  # noqa: E402

_log = get_logger(__name__)


EXTRACTIONS_ROOT = ROOT / "outputs" / "finetune" / "extractions"
LOGS_DIR = ROOT / "logs"


def _output_path(ref_id: str, start: str, end: str, tile: str) -> Path:
    """Mirror ``extract_local.extract``'s output naming."""
    return (EXTRACTIONS_ROOT / ref_id /
            f"WORLDCEREAL_{ref_id}_{start}_{end}_{tile}_LOCAL.geoparquet")


def _run_one_tile(args: dict) -> dict:
    tile = args["tile"]
    out_path = _output_path(args["ref_id"], args["start"], args["end"], tile)
    if out_path.exists() and not args["force"]:
        return {"tile": tile, "status": "skipped_exists", "elapsed": 0,
                "out": str(out_path)}

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"extract_{args['ref_id']}_{tile}.log"

    cmd = [
        args["python"], str(ROOT / "scripts" / "extract_local.py"), "extract",
        "--samples", args["samples"],
        "--ref-id", args["ref_id"],
        "--start-date", args["start"],
        "--end-date", args["end"],
        "--tile", tile,
        "--cloud-cover-max", str(args["cloud_cover_max"]),
        "--s2-max-scenes-per-month", str(args["s2_cap"]),
        "--s1-max-scenes-per-month", str(args["s1_cap"]),
    ]
    t0 = time.time()
    with open(log_path, "w") as logf:
        proc = subprocess.run(
            cmd, stdout=logf, stderr=subprocess.STDOUT,
            cwd=str(ROOT),
        )
    elapsed = time.time() - t0
    return {
        "tile": tile,
        "status": "ok" if proc.returncode == 0 else f"fail_rc{proc.returncode}",
        "elapsed": elapsed,
        "log": str(log_path),
        "out": str(out_path) if out_path.exists() else None,
    }


@click.command()
@click.option("--tile-dir", required=True, type=click.Path(exists=True))
@click.option("--ref-id", required=True)
@click.option("--start-date", "start", required=True)
@click.option("--end-date", "end", required=True)
@click.option("--min-samples-per-tile", default=30, show_default=True, type=int)
@click.option("--parallel", default=4, show_default=True, type=int)
@click.option("--cloud-cover-max", default=60, show_default=True, type=int)
@click.option("--s2-cap", default=4, show_default=True, type=int,
              help="Top-N cleanest S2 scenes per month.")
@click.option("--s1-cap", default=6, show_default=True, type=int,
              help="First-N S1 scenes per month.")
@click.option(
    "--python",
    default="/home/prog/PycharmProjects/ai-train-on-gis-data/.venv/bin/python",
    show_default=True,
    help="Interpreter that has worldcereal+rioxarray+pystac_client installed.",
)
@click.option("--force", is_flag=True, default=False,
              help="Re-run even if output parquet already exists.")
def main(
    tile_dir: str, ref_id: str, start: str, end: str,
    min_samples_per_tile: int, parallel: int, cloud_cover_max: int,
    s2_cap: int, s1_cap: int,
    python: str, force: bool,
) -> None:
    manifest_path = Path(tile_dir) / "manifest.json"
    summary = json.loads(manifest_path.read_text())
    spt = summary["samples_per_tile"]

    tiles_to_run = [t for t, n in spt.items() if n >= min_samples_per_tile]
    skipped = sorted(t for t, n in spt.items() if n < min_samples_per_tile)
    total_run = sum(spt[t] for t in tiles_to_run)
    total_skipped = sum(spt[t] for t in skipped)

    _log.info("manifest: %d tiles, %d samples total",
              len(spt), summary["total_samples"])
    _log.info("running:  %d tiles, %d samples  (>=%d/tile)",
              len(tiles_to_run), total_run, min_samples_per_tile)
    if skipped:
        _log.info("skipping: %d tiles, %d samples (<%d/tile): %s",
                  len(skipped), total_skipped, min_samples_per_tile,
                  ", ".join(skipped))

    # Largest tiles first so they don't tail-block the schedule.
    tiles_to_run.sort(key=lambda t: -spt[t])
    jobs = [
        {
            "tile": t,
            "samples": str(Path(tile_dir) / f"samples_v3_{t}.geoparquet"),
            "ref_id": ref_id,
            "start": start, "end": end,
            "cloud_cover_max": cloud_cover_max,
            "s2_cap": s2_cap,
            "s1_cap": s1_cap,
            "python": python,
            "force": force,
        }
        for t in tiles_to_run
    ]

    t0 = time.time()
    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=parallel) as pool:
        futures = {pool.submit(_run_one_tile, j): j["tile"] for j in jobs}
        done = 0
        for fut in as_completed(futures):
            r = fut.result()
            done += 1
            results.append(r)
            _log.info("[%d/%d] %s -> %s (%.0fs)",
                      done, len(jobs), r["tile"], r["status"], r["elapsed"])

    elapsed = time.time() - t0
    ok = sum(1 for r in results if r["status"] == "ok")
    skipped_existed = sum(1 for r in results if r["status"] == "skipped_exists")
    fail = [r for r in results if r["status"].startswith("fail")]
    _log.info("done in %.1f min: ok=%d skipped_existed=%d fail=%d",
              elapsed / 60, ok, skipped_existed, len(fail))
    if fail:
        _log.warning("failed tiles: %s", ", ".join(r["tile"] for r in fail))

    summary_out = {
        "ref_id": ref_id,
        "elapsed_minutes": round(elapsed / 60, 2),
        "ok": ok, "skipped_existed": skipped_existed,
        "fail": [r["tile"] for r in fail],
        "results": results,
    }
    out = EXTRACTIONS_ROOT / ref_id / "extraction_run_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary_out, indent=2))
    _log.info("wrote %s", out)


if __name__ == "__main__":
    main()
