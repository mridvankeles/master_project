# The MoE never routed. Every null result so far is uninformative.

> **Date:** 2026-08-12 · measured by `scripts/analyse_routing.py`
> **Supersedes the interpretation** in `results-full-scale-and-moe.md` §4 and
> `results-night-nwd-and-3expert.md` §2. The numbers there stand; what they mean
> does not.

## What was measured

Two questions that mAP cannot answer, and that were never asked until now:

1. Does the gate route by condition?
2. Do the experts compute different functions?

The union corpus encodes the true condition in every filename, so the first is a
direct comparison. The second is a forward pass of every expert on the same
input.

## Result

| Model | route distribution (eval) | NMI(route ; condition) | inter-expert CKA |
|---|---|---:|---:|
| MoE ×2 @ neck | e0 0.000 / e1 **1.000** | **0.0000** | **0.996** |
| MoE ×2 @ backbone | e0 0.081 / e1 0.919 | **0.0000** | — |
| MoE ×3 @ neck | e0 0.000 / e1 **1.000** / e2 0.000 | **0.0000** | 0.968–0.973 |
| MoE ×2 @ neck + NWD | e0 0.000 / e1 **1.000** | **0.0000** | **0.993** |

Both failure modes are present simultaneously:

- **The gate carries zero information about the condition.** At inference every
  image takes the same route. In the backbone model, where 8.1% take the
  minority route, those split evenly between clear (36) and fog (37) — the
  choice is uncorrelated with the condition it was supposed to detect.
- **The experts are clones anyway.** CKA 0.97–0.996 between every pair. Even if
  the gate had routed perfectly, both branches compute nearly the same function,
  so the route would not change the output.

## The reporting error that hid it

Training logs showed healthy balance — 0.375/0.625 for two experts, 0.240/0.438/
0.323 for three — and those were used to claim "no collapse". They were
measuring the wrong thing.

`MoEBlock.forward` adds Gaussian noise to the gate logits during training and
takes the argmax of the *noisy* logits. `last_index` therefore records the noisy
route, and the utilisation logs describe the injected noise rather than the
router. Noise was doing 100% of the routing diversity; remove it at eval and the
gate collapses to one expert.

The noisy top-k fix (added after observing a genuine collapse to a 0.000 share)
addressed the symptom in the logs and not the cause. **`last_index` must record
the clean argmax, with the noisy index kept separately.**

## What this invalidates

Every MoE-vs-dense comparison in the two results documents compared a dense
model against **a dense model with a redundant, never-selected branch**. That
comparison is trivially expected to show no gain, and it does (−0.002 to −0.006
mAP50, consistently).

So the conclusion "routing does not help here" was never tested. What was tested
is "an unused branch does not help", which says nothing about routing.

The interference results (§2 of both documents) are unaffected — they involve no
MoE. The NWD results are unaffected. **The MoE conclusions are withdrawn.**

## Why it collapsed — four contributing causes

1. **Zero-initialised experts.** Introduced so the block is an exact identity at
   step 0 and can be dropped into pretrained weights losslessly. The side effect
   is that both experts start as the same function, and nothing breaks the tie.
2. **No supervision on the gate.** The entropy auxiliary loss asks only for
   *balance*, never for *meaning*. A gate that splits images 50/50 on brightness
   quantisation satisfies it exactly as well as one that splits on condition.
3. **The residual path dominates.** The block returns `proj(x) + expert(x)` with
   bottlenecked experts, so swapping experts changes the output very little —
   which means very little gradient pressure to make them differ.
4. **Nothing penalised expert similarity.** `06-moe-design-guide.md` §2.5 lists
   SimSMoE for precisely this, and it was not used.

## What this means for the thesis

The MoE hypothesis is **untested**, not refuted. The design goal — one branch
specialising on haze, another on low light, selected by a gate — has not been
attempted in a form capable of exhibiting it, because the branches never
diverged and the gate never selected.

Three things must be true before any MoE result is interpretable, and all three
are checkable by `analyse_routing.py`:

- NMI(route ; condition) meaningfully above 0
- inter-expert CKA meaningfully below 1
- eval-time route distribution not concentrated on one expert

**These are now the acceptance criteria for the MoE arm**, and they should be
checked before accuracy is even looked at.
