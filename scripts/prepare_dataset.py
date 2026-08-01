"""Convert DIOR HBB annotations and materialise the Ultralytics corpus.

    python scripts/prepare_dataset.py

Reads the read-only release, writes data/dior_hbb/ (hardlinked images + label
txts + manifest.csv) and configs/data/dior_{clear,fog}.yaml.

Idempotent: re-running overwrites in place.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.build_dataset import (  # noqa: E402
    CONDITIONS,
    SPLITS,
    build,
    write_data_yaml,
    write_manifest,
)
from src.utils.logging import get_logger  # noqa: E402
from src.utils.paths import CONFIG_DIR, dataset_root, ensure_dir, load_paths  # noqa: E402

log = get_logger("prepare_dataset")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="path to paths.yaml")
    parser.add_argument("--out", default=None, help="dataset output root")
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=list(CONDITIONS),
        choices=list(CONDITIONS),
        help="which conditions to materialise",
    )
    args = parser.parse_args()

    paths = load_paths(args.config)
    problems = paths.validate()
    if problems:
        for p in problems:
            log.error(p)
        return 2

    out_root = ensure_dir(Path(args.out) if args.out else dataset_root())
    log.info("source: %s", paths.hazy_dior_root)
    log.info("target: %s", out_root)

    report = build(paths, out_root, conditions=tuple(args.conditions))

    counts = report.counts()
    for cond in args.conditions:
        row = counts[cond]
        log.info(
            "%-6s train=%-6s val=%-6s test=%-6s total=%s",
            cond,
            f"{row['train']:,}",
            f"{row['val']:,}",
            f"{row['test']:,}",
            f"{sum(row.values()):,}",
        )

    log.info("images: %s hardlinked, %s copied", f"{report.linked:,}", f"{report.copied:,}")

    if report.issues:
        kinds = Counter(i.kind for i in report.issues)
        log.warning("geometry issues recorded: %s", dict(kinds))
        for issue in report.issues[:10]:
            log.warning("  %s[%s] %s: %s", issue.image_id, issue.index, issue.kind, issue.detail)
        if len(report.issues) > 10:
            log.warning("  ... and %s more", len(report.issues) - 10)
    else:
        log.info("no out-of-bounds / zero-area / size-mismatch boxes found")

    if report.missing_sources:
        log.error("%s missing sources:", len(report.missing_sources))
        for m in report.missing_sources[:10]:
            log.error("  %s", m)

    manifest = write_manifest(report, out_root)
    log.info("wrote %s (%s rows)", manifest, f"{len(report.rows):,}")

    for cond in args.conditions:
        dst = write_data_yaml(out_root, cond, CONFIG_DIR / "data" / f"dior_{cond}.yaml")
        log.info("wrote %s", dst)

    empty = [
        f"{c}/{s}" for c in args.conditions for s in SPLITS if counts[c][s] == 0
    ]
    if empty:
        log.error("empty splits: %s", ", ".join(empty))
        return 1

    return 1 if report.missing_sources else 0


if __name__ == "__main__":
    raise SystemExit(main())
