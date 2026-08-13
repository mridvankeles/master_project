"""Does the fog branch fire on REAL haze, or only on our synthesiser?

RRSHID is real-world paired clear/hazy remote-sensing imagery. It carries no
detection labels, so it cannot train or score a detector -- but the router needs
no labels at all. Running the trained gate on real hazy images and their own
clear counterparts is a direct test of whether it learned "haze" or "the
parameters of our haze simulator", which is the domain-shift question
04-method-open-questions.md calls the core result of the analysis chapter.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, "c:/Users/Ridvan/Desktop/tez/MasterRepoistory")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.models.moe2 import cond_moe_blocks  # noqa: E402
from ultralytics import YOLO  # noqa: E402

RRSHID = Path("C:/Users/Ridvan/Desktop/tez/RRSHID")
RUN = "outputs/runs/cond3_gated_yolo11n/weights/best.pt"

net = YOLO(f"c:/Users/Ridvan/Desktop/tez/MasterRepoistory/{RUN}").model.cuda().eval().float()
blk = cond_moe_blocks(net)[0]
names = list(blk.expert_kinds)


def gate_on(paths, limit=160):
    rows = []
    with torch.no_grad():
        for i in range(0, min(len(paths), limit), 16):
            ch = paths[i : i + 16]
            ims = []
            for p in ch:
                im = cv2.imread(str(p))
                if im is None:
                    continue
                ims.append(
                    cv2.cvtColor(cv2.resize(im, (640, 640)), cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
                )
            if not ims:
                continue
            x = torch.from_numpy(np.stack(ims)).float().div(255).cuda()
            net(x)
            rows.append(blk.last_gate.cpu().numpy())
    return np.concatenate(rows) if rows else np.zeros((0, len(names)))


print(f"{'source':34s}" + "".join(f"{n:>10s}" for n in names) + f"{'argmax=fog':>12s}")
results = {}
for sev in ("moderate_fog", "thick_fog"):
    for kind in ("clear", "hazy"):
        ps = sorted((RRSHID / sev / "test" / kind).glob("*.png"))
        if not ps:
            continue
        P = gate_on(ps)
        results[f"{sev}/{kind}"] = P
        frac_fog = float((P.argmax(1) == names.index("fog")).mean())
        print(
            f"REAL {sev}/{kind:6s} (n={len(P):3d})".ljust(34)
            + "".join(f"{v:10.3f}" for v in P.mean(0))
            + f"{frac_fog:12.3f}"
        )

# Same measurement on our own synthetic corpus, for a like-for-like comparison.
SYN = Path("c:/Users/Ridvan/Desktop/tez/MasterRepoistory/data/dior_hbb_full")
for cond, sub in (("clear", SYN / "clear/images/val"), ("fog", SYN / "fog/images/val")):
    ps = sorted(sub.iterdir())[:160]
    P = gate_on(ps)
    frac_fog = float((P.argmax(1) == names.index("fog")).mean())
    print(
        f"SYNTH {cond:12s} (n={len(P):3d})".ljust(34)
        + "".join(f"{v:10.3f}" for v in P.mean(0))
        + f"{frac_fog:12.3f}"
    )

print("\nfog-branch probability, real hazy MINUS its own real clear:")
for sev in ("moderate_fog", "thick_fog"):
    a, b = results.get(f"{sev}/hazy"), results.get(f"{sev}/clear")
    if a is not None and b is not None and len(a) and len(b):
        i = names.index("fog")
        print(f"  {sev:14s} {a[:, i].mean() - b[:, i].mean():+.4f}"
              f"   (hazy {a[:, i].mean():.3f} vs clear {b[:, i].mean():.3f})")
