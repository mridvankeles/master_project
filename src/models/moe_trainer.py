"""A `DetectionTrainer` that adds the MoE auxiliary loss and logs utilisation.

WHERE THE HOOK IS
-----------------
Ultralytics 8.4's training loop calls `self.model(batch)`, which routes through
`BaseModel.forward(dict)` -> `BaseModel.loss(batch)` -> `self.criterion(...)`.
The clean interception point is therefore the model's bound `loss`, wrapped once
in `get_model`. Overriding a `criterion` method would not work: the trainer
reassigns `model.criterion` during setup, so the override would be discarded.

TWO RESPONSIBILITIES
--------------------
1. **Auxiliary loss.** Added with weight `moe_lambda`. Without it the gate may
   collapse onto one expert and the block degenerates into a dense model
   carrying a dead branch.
2. **Utilisation logging.** `06-moe-design-guide.md` §3.3: a utilisation
   histogram is the only way to see collapse happening. Shares are accumulated
   per epoch and pushed to MLflow, so a dead expert shows up in the run rather
   than being inferred later from a disappointing number.

The auxiliary value is logged separately from the detection loss — folded
together, a routing problem and a detection problem look identical on the curve.
"""

from __future__ import annotations

from collections import defaultdict

import torch
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import DEFAULT_CFG, LOGGER

from .moe import moe_blocks, routing_aux_loss


class MoEDetectionTrainer(DetectionTrainer):
    """DetectionTrainer + routing auxiliary loss.

    Configure via `overrides`:
        moe_lambda: weight on the auxiliary term (0 disables it)
        moe_aux:    "entropy" | "cv" | "switch"
    """

    # `cfg=DEFAULT_CFG`, not None: Ultralytics instantiates a custom trainer as
    # `trainer(overrides=args, _callbacks=...)` with no cfg, so a None default
    # would reach get_cfg and fail on `.keys()`.
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        overrides = dict(overrides or {})
        self.moe_lambda = float(overrides.pop("moe_lambda", 0.01))
        self.moe_aux = str(overrides.pop("moe_aux", "entropy"))
        self._epoch_shares: dict[str, float] = defaultdict(float)
        self._epoch_batches = 0
        self._last_aux = 0.0
        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)
        self.add_callback("on_fit_epoch_end", _log_utilisation)

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
        blocks = moe_blocks(model)
        LOGGER.info(
            f"MoE trainer: {len(blocks)} block(s), lambda={self.moe_lambda}, aux={self.moe_aux}"
        )
        if self.moe_lambda == 0 or not blocks:
            LOGGER.warning("MoE trainer: auxiliary loss NOT installed")
            return model

        inner_loss = model.loss  # bound method; instance attribute will shadow it

        def loss_with_aux(batch, preds=None):
            total, items = inner_loss(batch, preds)
            aux = routing_aux_loss(model, mode=self.moe_aux)
            if isinstance(aux, torch.Tensor):
                self._last_aux = float(aux.detach())
                term = self.moe_lambda * aux
                # The loop does `self.loss = loss.sum()`, so a scalar added to a
                # vector-valued loss would be counted once per component. Add it
                # to a single element instead, leaving `items` (what gets
                # displayed as box/cls/dfl) untouched.
                if isinstance(total, torch.Tensor) and total.ndim > 0:
                    total = total.clone()
                    total[0] = total[0] + term
                else:
                    total = total + term
            self._accumulate(model)
            return total, items

        model.loss = loss_with_aux
        return model

    def save_model(self):
        """Save with the loss wrapper detached.

        `get_model` installs a closure as an INSTANCE attribute on the model so
        it shadows `BaseModel.loss`. torch.save pickles the model object, and a
        local closure is not picklable, so every checkpoint write would fail.
        The wrapper is therefore removed for the duration of the save and put
        back afterwards — on the EMA copy too, since ModelEMA deepcopies the
        model and inherits the same attribute.
        """
        stashed = []
        for obj in (getattr(self, "model", None), getattr(getattr(self, "ema", None), "ema", None)):
            if obj is not None and "loss" in getattr(obj, "__dict__", {}):
                stashed.append((obj, obj.__dict__.pop("loss")))
        try:
            return super().save_model()
        finally:
            for obj, fn in stashed:
                obj.__dict__["loss"] = fn

    def _accumulate(self, model) -> None:
        for b_i, block in enumerate(moe_blocks(model)):
            idx = block.last_index
            if idx is None:
                continue
            for e in range(block.n_experts):
                self._epoch_shares[f"moe{b_i}/expert{e}_share"] += float(
                    (idx == e).sum()
                ) / max(idx.numel(), 1)
        self._epoch_batches += 1

    def epoch_utilisation(self) -> dict[str, float]:
        """Mean per-expert share over the epoch, then reset the accumulator."""
        if not self._epoch_batches:
            return {}
        out = {k: v / self._epoch_batches for k, v in self._epoch_shares.items()}
        out["moe/aux_loss"] = self._last_aux
        # min share across experts: one number that says "is anything dying?"
        shares = [v for k, v in out.items() if k.endswith("_share")]
        if shares:
            out["moe/min_expert_share"] = min(shares)
        self._epoch_shares.clear()
        self._epoch_batches = 0
        return out


def _log_utilisation(trainer) -> None:
    stats = trainer.epoch_utilisation() if hasattr(trainer, "epoch_utilisation") else {}
    if not stats:
        LOGGER.warning("MoE: no routing statistics this epoch (loss wrapper not reached?)")
        return
    # To the console as well as MLflow: a dying expert should be visible while
    # the run is happening, not only in a dashboard afterwards.
    LOGGER.info("MoE utilisation: " + ", ".join(f"{k}={v:.3f}" for k, v in sorted(stats.items())))
    try:
        import mlflow

        if mlflow.active_run() is not None:
            mlflow.log_metrics(stats, step=trainer.epoch)
    except Exception:  # noqa: BLE001 - logging must never kill a training run
        pass
