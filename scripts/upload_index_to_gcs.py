"""Upload the FAISS index to GCS bucket `index_save`.

Usage:
    # Auth first (one of):
    #   gcloud auth application-default login
    #   export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
    #
    # Then:
    python scripts/upload_index_to_gcs.py
    python scripts/upload_index_to_gcs.py --bucket index_save --src /path/to/file --dest some/object/name
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from google.cloud import storage
from google.cloud.storage import transfer_manager

DEFAULT_SRC = "/workspace/reason_over_search/local_retriever/indexes/wiki18_100w_e5_ivf4096_sq8.index"
DEFAULT_BUCKET = "index_save"


def upload_with_transfer_manager(
    bucket_name: str,
    src_path: Path,
    dest_blob_name: str,
    chunk_size_mib: int = 64,
    workers: int = 8,
) -> None:
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(dest_blob_name)

    size_bytes = src_path.stat().st_size
    print(f"Uploading {src_path} ({size_bytes / 1024**3:.2f} GiB) "
          f"to gs://{bucket_name}/{dest_blob_name}")
    print(f"Chunk size: {chunk_size_mib} MiB, workers: {workers}")

    start = time.time()
    transfer_manager.upload_chunks_concurrently(
        str(src_path),
        blob,
        chunk_size=chunk_size_mib * 1024 * 1024,
        max_workers=workers,
    )
    elapsed = time.time() - start
    mibps = (size_bytes / 1024**2) / max(elapsed, 1e-6)
    print(f"Done in {elapsed:.1f}s ({mibps:.1f} MiB/s)")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=DEFAULT_SRC)
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument(
        "--dest",
        default=None,
        help="Destination object name in the bucket (default: source filename)",
    )
    parser.add_argument("--chunk-size-mib", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    src_path = Path(args.src)
    if not src_path.is_file():
        print(f"ERROR: source file not found: {src_path}", file=sys.stderr)
        return 2

    dest_blob_name = args.dest or src_path.name

    upload_with_transfer_manager(
        bucket_name=args.bucket,
        src_path=src_path,
        dest_blob_name=dest_blob_name,
        chunk_size_mib=args.chunk_size_mib,
        workers=args.workers,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
