"""Download the full WorldCereal public extractions dataset (Presto fine-tune corpus).

Source: ``https://s3.waw3-1.cloudferro.com/swift/v1/geoparquet/`` (Cloudferro
public S3 bucket — same one ``worldcereal.utils.refdata.query_public_extractions``
queries with DuckDB). The bucket is unauthenticated; just plain HTTPS GETs.

Layout written locally::

    <out_dir>/
        worldcereal_public_extractions_extent.parquet
        worldcereal_public_extractions.parquet/
            ref_id=2017_AF_One-Acre-Fund-MEL_POINT_110/...parquet
            ref_id=2017_BEL_LPIS-Flanders_POLY_110/...parquet
            ...

The directory mirrors the bucket key prefix so DuckDB / pandas can read the
parquet partitions directly via ``read_parquet(out_dir + '**/*.parquet')``.

Usage
-----
    .venv/bin/python scripts/download_worldcereal_finetune.py \\
        --out-dir data/worldcereal_public_extractions \\
        --workers 8

Resume: existing files whose size matches the remote ``Content-Length`` are
skipped. Mismatches are re-downloaded.
"""

from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import click
import requests
from tqdm import tqdm


LISTING_URL = "https://s3.waw3-1.cloudferro.com/swift/v1/geoparquet/"
PREFIX = "worldcereal_public_extractions"


def list_remote_files(session: requests.Session, timeout: int) -> list[str]:
    r = session.get(LISTING_URL, timeout=timeout)
    r.raise_for_status()
    keys = [
        line.strip()
        for line in r.text.split("\n")
        if line.strip().endswith(".parquet") and line.strip().startswith(PREFIX)
    ]
    if not keys:
        raise click.ClickException(
            f"no parquet keys returned from {LISTING_URL} — bucket layout changed?"
        )
    return keys


def remote_size(session: requests.Session, key: str, timeout: int) -> int:
    h = session.head(LISTING_URL + key, timeout=timeout, allow_redirects=True)
    h.raise_for_status()
    cl = h.headers.get("content-length")
    return int(cl) if cl is not None else -1


def download_one(
    session: requests.Session,
    key: str,
    out_dir: Path,
    timeout: int,
    chunk_size: int,
    retries: int,
) -> tuple[str, int, str]:
    """Return (key, bytes_written, status). Status: 'downloaded' | 'skipped' | 'error:<msg>'."""
    target = out_dir / key
    target.parent.mkdir(parents=True, exist_ok=True)

    try:
        expected = remote_size(session, key, timeout)
    except Exception as exc:  # noqa: BLE001
        return key, 0, f"error:HEAD failed: {exc}"

    if target.exists() and expected > 0 and target.stat().st_size == expected:
        return key, 0, "skipped"

    tmp = target.with_suffix(target.suffix + ".partial")
    last_err = ""
    for attempt in range(1, retries + 1):
        try:
            with session.get(LISTING_URL + key, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                written = 0
                with tmp.open("wb") as fh:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:
                            fh.write(chunk)
                            written += len(chunk)
            if expected > 0 and written != expected:
                last_err = f"size mismatch: got {written}, expected {expected}"
                tmp.unlink(missing_ok=True)
                continue
            tmp.replace(target)
            return key, written, "downloaded"
        except Exception as exc:  # noqa: BLE001
            last_err = f"{type(exc).__name__}: {exc}"
            tmp.unlink(missing_ok=True)
            time.sleep(min(2 ** attempt, 30))

    return key, 0, f"error:{last_err}"


@click.command()
@click.option(
    "--out-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("data/worldcereal_public_extractions"),
    show_default=True,
    help="Local directory to mirror the bucket prefix into.",
)
@click.option("--workers", type=int, default=8, show_default=True,
              help="Parallel HTTP downloads.")
@click.option("--timeout", type=int, default=120, show_default=True,
              help="Per-request timeout in seconds.")
@click.option("--chunk-size", type=int, default=1 << 20, show_default=True,
              help="Streaming chunk size in bytes (default 1 MiB).")
@click.option("--retries", type=int, default=4, show_default=True,
              help="Per-file retry attempts on failure.")
@click.option("--include", default=None,
              help="Optional substring filter on ref_id keys (e.g. '2024_').")
@click.option("--list-only", is_flag=True,
              help="Print remote files + sizes and exit; do not download.")
def main(
    out_dir: Path,
    workers: int,
    timeout: int,
    chunk_size: int,
    retries: int,
    include: str | None,
    list_only: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": "ai-train-on-gis-data/wc-downloader"})

    click.echo(f"Listing {LISTING_URL}")
    keys = list_remote_files(session, timeout)
    if include:
        keys = [k for k in keys if include in k]
    click.echo(f"  {len(keys)} parquet files matched")

    if list_only:
        for k in keys:
            sz = remote_size(session, k, timeout)
            click.echo(f"  {sz:>12d}  {k}")
        return

    n_done = n_skip = n_err = 0
    bytes_written = 0
    errors: list[tuple[str, str]] = []

    pbar = tqdm(total=len(keys), desc="downloading", unit="file")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(
                download_one, session, k, out_dir,
                timeout, chunk_size, retries,
            ): k
            for k in keys
        }
        for fut in as_completed(futs):
            key, written, status = fut.result()
            if status == "downloaded":
                n_done += 1
                bytes_written += written
            elif status == "skipped":
                n_skip += 1
            else:
                n_err += 1
                errors.append((key, status))
            pbar.set_postfix(
                ok=n_done, skip=n_skip, err=n_err,
                gb=f"{bytes_written/1e9:.2f}",
            )
            pbar.update(1)
    pbar.close()

    click.echo(
        f"\ndone: downloaded={n_done}  skipped={n_skip}  errors={n_err}  "
        f"new_bytes={bytes_written/1e9:.2f} GB"
    )
    if errors:
        click.echo("\nerrors:")
        for k, msg in errors:
            click.echo(f"  {k}\n    {msg}")
        sys.exit(1)


if __name__ == "__main__":
    main()
