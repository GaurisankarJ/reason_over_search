"""Generate a v4 signed download URL for an object in GCS.

Note: GCS signed URLs are time-limited, not single-use. Anyone holding the URL
can download the object any number of times until it expires (max 7 days).
Treat the URL itself as a credential.

Usage:
    export GOOGLE_APPLICATION_CREDENTIALS=/path/to/sa.json
    python scripts/generate_signed_url.py
    python scripts/generate_signed_url.py --hours 24
    python scripts/generate_signed_url.py --bucket index_save --object some/path.bin --hours 1
"""

from __future__ import annotations

import argparse
import sys
from datetime import timedelta

from google.cloud import storage

DEFAULT_BUCKET = "index_save"
DEFAULT_OBJECT = "wiki18_100w_e5_ivf4096_sq8.index"
MAX_HOURS = 7 * 24  # GCS v4 signed URL upper bound


def generate(bucket_name: str, object_name: str, hours: float) -> str:
    if hours <= 0 or hours > MAX_HOURS:
        raise ValueError(f"--hours must be in (0, {MAX_HOURS}]")
    client = storage.Client()
    blob = client.bucket(bucket_name).blob(object_name)
    if not blob.exists():
        raise FileNotFoundError(f"gs://{bucket_name}/{object_name} not found")
    return blob.generate_signed_url(
        version="v4",
        expiration=timedelta(hours=hours),
        method="GET",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--object", default=DEFAULT_OBJECT, dest="object_name")
    parser.add_argument(
        "--hours",
        type=float,
        default=24.0,
        help=f"Lifetime in hours (max {MAX_HOURS}). Default 24.",
    )
    args = parser.parse_args()

    url = generate(args.bucket, args.object_name, args.hours)
    print(f"# gs://{args.bucket}/{args.object_name}")
    print(f"# expires in {args.hours}h")
    print(url)
    print()
    print("# Recipient downloads with:")
    print(f"#   curl -L -o {args.object_name.split('/')[-1]} '<URL>'")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise SystemExit(2)
