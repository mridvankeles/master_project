"""Register `MoEBlock` inside the vendored, editable Ultralytics clone.

    python scripts/install_moe.py            # patch
    python scripts/install_moe.py --check    # verify only, exit 1 if missing

WHY A PATCH SCRIPT
------------------
`ultralytics/nn/tasks.py` resolves a YAML module name through `globals()[m]`,
so a custom block has to be importable from that module's namespace. Editing
the clone by hand works but is invisible to version control -- `.gitignore`
excludes `/ultralytics-src/`, and `make setup` re-clones it at a pinned SHA, so
a hand edit is silently lost on the other machine. This script is the edit,
committed, idempotent, and re-runnable after any re-clone.

The block itself lives in `src/models/moe.py` in this repo. Ultralytics only
gets an import and two registrations, so there is one definition of the model.

WHAT IT CHANGES (three lines, all idempotent)
  1. `ultralytics/nn/modules/__init__.py` -- import + __all__ entry
  2. `ultralytics/nn/tasks.py`            -- import into the parse_model namespace
  3. `ultralytics/nn/tasks.py`            -- add MoEBlock to `base_modules`

(3) is what makes parse_model pass `(c1, c2_scaled, *args)` instead of the raw
YAML args, so the block picks up input channels and width scaling like every
built-in block does.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.logging import get_logger  # noqa: E402
from src.utils.paths import REPO_ROOT  # noqa: E402

log = get_logger("install_moe")

# `make setup` clones here; keep in sync with the Makefile's ULTRALYTICS_DIR.
DEFAULT_CLONE = REPO_ROOT.parent / "ultralytics-src"

IMPORT_LINE = (
    "\n# --- thesis MoE block (registered by scripts/install_moe.py) ---------------\n"
    "import sys as _sys\n"
    "from pathlib import Path as _Path\n"
    "_repo = _Path(__file__).resolve().parents[3] / {repo!r}\n"
    "if str(_repo) not in _sys.path:\n"
    "    _sys.path.insert(0, str(_repo))\n"
    "from src.models.moe import MoEBlock  # noqa: E402,F401\n"
)

MARKER = "thesis MoE block (registered by scripts/install_moe.py)"


def _patch_modules_init(path: Path, repo: str) -> bool:
    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        return False
    text += IMPORT_LINE.format(repo=repo)
    if '"MoEBlock"' not in text:
        text += '\n__all__ = tuple(__all__) + ("MoEBlock",)\n'
    path.write_text(text, encoding="utf-8")
    return True


def _patch_tasks(path: Path, repo: str) -> bool:
    text = path.read_text(encoding="utf-8")
    changed = False

    if MARKER not in text:
        text += IMPORT_LINE.format(repo=repo)
        changed = True

    # Add to base_modules so parse_model injects (c1, c2) and applies width
    # scaling. Without this the block lands in the `else: c2 = ch[f]` branch and
    # receives the raw YAML args, so it never learns its own input width.
    # The set literal is `base_modules = frozenset(\n        {\n ... \n        }\n    )`,
    # so the entry must go before the closing BRACE, not before the closing
    # paren -- inserting between `}` and `)` is a syntax error.
    anchor = "    base_modules = frozenset(\n        {\n"
    if anchor in text and "MoEBlock," not in text.split(anchor, 1)[1][:3000]:
        head, tail = text.split(anchor, 1)
        close = tail.index("        }\n")
        tail = tail[:close] + "            MoEBlock,\n" + tail[close:]
        text = head + anchor + tail
        changed = True

    if changed:
        path.write_text(text, encoding="utf-8")
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clone", default=None, help="path to the ultralytics clone")
    parser.add_argument("--check", action="store_true", help="verify only")
    args = parser.parse_args()

    clone = Path(args.clone) if args.clone else DEFAULT_CLONE
    tasks = clone / "ultralytics" / "nn" / "tasks.py"
    modules_init = clone / "ultralytics" / "nn" / "modules" / "__init__.py"
    if not tasks.exists():
        log.error("no Ultralytics clone at %s (run `make setup`)", clone)
        return 2

    if args.check:
        import importlib

        try:
            nn_tasks = importlib.import_module("ultralytics.nn.tasks")
            ok = hasattr(nn_tasks, "MoEBlock")
        except Exception as exc:  # noqa: BLE001
            log.error("import failed: %s", exc)
            return 1
        log.info("MoEBlock visible to parse_model: %s", ok)
        return 0 if ok else 1

    repo = REPO_ROOT.name
    a = _patch_modules_init(modules_init, repo)
    b = _patch_tasks(tasks, repo)
    log.info("modules/__init__.py: %s", "patched" if a else "already registered")
    log.info("tasks.py           : %s", "patched" if b else "already registered")

    # Prove it, rather than trusting the string edit.
    import importlib

    for name in ("ultralytics.nn.modules", "ultralytics.nn.tasks", "ultralytics"):
        sys.modules.pop(name, None)
    nn_tasks = importlib.import_module("ultralytics.nn.tasks")
    if not hasattr(nn_tasks, "MoEBlock"):
        log.error("MoEBlock still not visible to parse_model")
        return 1
    log.info("verified: parse_model can resolve 'MoEBlock'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
