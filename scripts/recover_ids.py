"""Recover DIOR ids + severities for the renumbered Hazy-DIOR subtrees.

    python scripts/recover_ids.py                 # build the mapping
    python scripts/recover_ids.py --verify-only   # re-check a cached mapping

Writes `data/hazy_dior_id_map.json`, which `prepare_dataset.py --scope full`
consumes. This takes ~25 minutes over 62,568 images, so it is a cached artefact
rather than something the build re-runs. See `src/data/recovery.py` for why the
mapping is recoverable at all and what is verified about it.

Exits non-zero if a verification check fails — the mapping decides which
annotation is attached to which image, so a silent failure here mislabels the
entire fog corpus.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.recovery import (  # noqa: E402
    load_mapping,
    recover,
    verify_recovery,
    write_mapping,
)
from src.utils.logging import get_logger  # noqa: E402
from src.utils.paths import DATA_DIR, load_paths  # noqa: E402

log = get_logger("recover_ids")

DEFAULT_OUT = DATA_DIR / "hazy_dior_id_map.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="path to paths.yaml")
    parser.add_argument("--out", default=None, help="where to write the mapping")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="re-run the verification checks against an existing mapping",
    )
    args = parser.parse_args()

    paths = load_paths(args.config)
    if paths.dior_root is None:
        log.error("dior_root is not set in paths.yaml — recovery needs the clear DIOR release")
        return 2
    problems = paths.validate()
    if problems:
        for p in problems:
            log.error(p)
        return 2

    out = Path(args.out) if args.out else DEFAULT_OUT

    if args.verify_only:
        if not out.exists():
            log.error("no mapping at %s — run without --verify-only first", out)
            return 2
        report = load_mapping(out)
        log.info("loaded %s entries from %s", f"{len(report.mapping):,}", out)
    else:
        def progress(n: int, total: int, label: str = "dior index") -> None:
            log.info("  %s %s/%s", label, f"{n:,}", f"{total:,}")

        log.info("recovering — this hashes ~86k images and takes roughly 25 minutes")
        report = recover(paths, progress=progress)

        for name, stats in report.stats.items():
            log.info("%-11s %s", name, stats)
        for problem in report.problems:
            log.warning("%s", problem)

        write_mapping(report, out)
        log.info("wrote %s (%s entries)", out, f"{len(report.mapping):,}")

    log.info("recovered %s unique DIOR ids", f"{len(report.recovered_ids):,}")

    failed = False
    for check in verify_recovery(paths, report):
        (log.info if check.passed else log.error)(
            "[%s] %s — %s", "PASS" if check.passed else "FAIL", check.name, check.detail
        )
        failed |= not check.passed

    if failed:
        log.error("verification failed — do NOT build a corpus from this mapping")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
