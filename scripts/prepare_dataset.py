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
    SCOPES,
    SPLITS,
    TASKS,
    build,
    write_data_yaml,
    write_manifest,
)
from src.data.recovery import load_mapping, sources_by_id  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.paths import (  # noqa: E402
    CONFIG_DIR,
    DATA_DIR,
    dataset_root,
    ensure_dir,
    load_paths,
)

DEFAULT_ID_MAP = DATA_DIR / "hazy_dior_id_map.json"

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
    parser.add_argument(
        "--task",
        default="detect",
        choices=list(TASKS),
        help="detect = horizontal boxes (default); obb = oriented",
    )
    parser.add_argument(
        "--scope",
        default="aligned",
        choices=list(SCOPES),
        help="aligned = the 2,607 DIOR-ID-keyed ids; full = all 23,463 "
             "(needs dior_root and the recovered id map)",
    )
    parser.add_argument(
        "--id-map",
        default=None,
        help="recovered id map for --scope full (default data/hazy_dior_id_map.json)",
    )
    args = parser.parse_args()

    paths = load_paths(args.config)
    problems = paths.validate()
    if problems:
        for p in problems:
            log.error(p)
        return 2

    out_root = ensure_dir(
        Path(args.out) if args.out else dataset_root(args.task, scope=args.scope)
    )

    id_map = None
    if args.scope == "full":
        if paths.dior_root is None:
            log.error("--scope full needs `dior_root` in configs/paths.yaml")
            return 2
        map_path = Path(args.id_map) if args.id_map else DEFAULT_ID_MAP
        if not map_path.exists():
            log.error("no id map at %s — run scripts/recover_ids.py first", map_path)
            return 2
        id_map = sources_by_id(load_mapping(map_path))
        log.info("id map: %s (%s ids with haze renders)", map_path, f"{len(id_map):,}")

    log.info("source: %s", paths.hazy_dior_root)
    if paths.dior_root:
        log.info("clear : %s", paths.dior_root)
    log.info("target: %s   task: %s   scope: %s", out_root, args.task, args.scope)

    report = build(
        paths,
        out_root,
        conditions=tuple(args.conditions),
        task=args.task,
        scope=args.scope,
        id_map=id_map,
    )

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

    if report.unrecoverable:
        # Bounded and expected: their clear image is pixel-identical to another
        # DIOR image, so recovery declines to guess which id the haze render
        # belongs to. Reported, counted, and not treated as a build failure.
        log.warning(
            "%s id(s) have no fog render (pixel-ambiguous in recovery); "
            "clear is still built for them", len(report.unrecoverable)
        )

    if report.missing_sources:
        log.error("%s missing sources:", len(report.missing_sources))
        for m in report.missing_sources[:10]:
            log.error("  %s", m)

    manifest = write_manifest(report, out_root)
    log.info("wrote %s (%s rows)", manifest, f"{len(report.rows):,}")

    fmts = report.format_counts()
    log.info("source container per condition: %s", fmts)
    if len({f for counts in fmts.values() for f in counts}) > 1:
        # Not fatal, but it must never be discovered late: see
        # BuildReport.format_counts for why this is a router confound.
        log.warning(
            "conditions differ in image container/compression history — record this "
            "in the thesis and check the router is not separating conditions on it"
        )

    suffix = "" if args.task == "detect" else "_obb"
    scope_suffix = "" if args.scope == "aligned" else f"_{args.scope}"
    for cond in args.conditions:
        dst = write_data_yaml(
            out_root,
            cond,
            CONFIG_DIR / "data" / f"dior_{cond}{suffix}{scope_suffix}.yaml",
            task=args.task,
        )
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
