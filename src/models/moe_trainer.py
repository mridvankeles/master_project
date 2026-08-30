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
        # Box-loss selection. Independent of the MoE: a run may use NWD with a
        # stock model, or an MoE with the stock box loss, and the two effects
        # have to be separable in the results table.
        # Weight on the supervised gate loss. This is what makes branch i MEAN
        # condition i; the entropy auxiliary only ever asked for balance, which
        # a gate splitting on anything at all satisfies equally well.
        self.gate_lambda = float(overrides.pop("gate_lambda", 0.0))
        # Calibration knobs for the gate. pos_weight rebalances the BCE;
        # count_lambda charges the gate for firing the wrong NUMBER of experts.
        self.gate_pos_weight = float(overrides.pop("gate_pos_weight", 1.0))
        self.gate_count_lambda = float(overrides.pop("gate_count_lambda", 0.0))
        self._last_count_loss = 0.0
        # Expert floor: charge every image that reaches no specialist at all.
        # Unlike count_lambda this needs no label, so it also governs OOD input.
        self.gate_floor_lambda = float(overrides.pop("gate_floor_lambda", 0.0))
        self.gate_floor_tau = float(overrides.pop("gate_floor_tau", 0.6))
        # Orthogonality on expert OUTPUTS (NeurIPS 2025). Cannot fix an inert
        # expert -- pair it with restore_lambda, never use it alone.
        self.gate_ortho_lambda = float(overrides.pop("gate_ortho_lambda", 0.0))
        self._last_ortho = 0.0
        self._last_floor_loss = 0.0
        self.nwd_mode = str(overrides.pop("nwd", "off"))
        self.nwd_c = float(overrides.pop("nwd_c", 12.8))
        self.nwd_tiny_area = float(overrides.pop("nwd_tiny_area", 32.0**2))
        # Paired restoration: teach the fog/night branches to reproduce the
        # block output of their CLEAR TWIN. Gives the experts an objective the
        # always-on branch cannot absorb, which is why design 2's were inert.
        self.restore_lambda = float(overrides.pop("restore_lambda", 0.0))
        self.restore_beta = float(overrides.pop("restore_beta", 0.1))
        self._last_restore = 0.0
        self.nwd_levels = str(overrides.pop("nwd_levels", "all"))
        self.nwd_p3_weight = float(overrides.pop("nwd_p3_weight", 1.0))
        self._nwd_installed = False
        self._epoch_shares: dict[str, float] = defaultdict(float)
        self._epoch_batches = 0
        self._last_aux = 0.0
        self._last_gate_loss = 0.0
        self._cond_report: dict[str, float] = {}
        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)
        self.add_callback("on_fit_epoch_end", _log_utilisation)

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
        from .moe2 import cond_moe_blocks

        blocks = moe_blocks(model)
        cond_blocks = cond_moe_blocks(model)
        LOGGER.info(
            f"MoE trainer: {len(blocks)} legacy block(s), {len(cond_blocks)} cond block(s), "
            f"lambda={self.moe_lambda}, aux={self.moe_aux}, "
            f"gate_lambda={self.gate_lambda}, floor={self.gate_floor_lambda}"
            f"@tau={self.gate_floor_tau}, nwd={self.nwd_mode}/{self.nwd_levels}"
            f" p3w={self.nwd_p3_weight}"
        )
        use_aux = self.moe_lambda != 0 and bool(blocks)
        gate_active = (self.gate_lambda or self.gate_floor_lambda
                       or self.gate_ortho_lambda or self.restore_lambda) and cond_blocks
        if not use_aux and self.nwd_mode == "off" and not gate_active:
            LOGGER.warning("MoE trainer: neither auxiliary loss nor NWD is active")
            return model

        inner_loss = model.loss  # bound method; instance attribute will shadow it

        def loss_with_aux(batch, preds=None):
            # The criterion is built lazily on the first loss call and is
            # reassigned during setup, so NWD can only be swapped in here --
            # doing it earlier would be silently undone.
            if not self._nwd_installed and self.nwd_mode != "off":
                from .nwd import install_nwd

                total_first, items_first = inner_loss(batch, preds)
                ok = install_nwd(model, c=self.nwd_c,
                                 tiny_area=self.nwd_tiny_area, mode=self.nwd_mode,
                                 levels=self.nwd_levels, p3_weight=self.nwd_p3_weight)
                self._nwd_installed = True
                LOGGER.info(f"NWD box loss installed: {ok} (mode={self.nwd_mode})")
                if not ok:
                    LOGGER.error("NWD requested but the criterion had no bbox_loss")
                total_first = self._add_gate_supervision(model, total_first, batch)
                total_first, items_first = self._finish(model, total_first, items_first, use_aux)
                return self._add_restoration(model, total_first, batch), items_first

            total, items = inner_loss(batch, preds)
            total = self._add_gate_supervision(model, total, batch)
            # _finish reads the block's routing state, so it must run BEFORE the
            # twin forward pass overwrites it.
            total, items = self._finish(model, total, items, use_aux)
            return self._add_restoration(model, total, batch), items

        model.loss = loss_with_aux
        return model

    def _add_gate_supervision(self, model, total, batch):
        """Cross-entropy between the gate and the condition labels in filenames.

        Training only. The validator calls `model.loss(batch, preds)` with preds
        already computed, so no forward re-runs and `last_logits` still holds a
        previous batch's gate output — supervising against the current batch's
        labels would then compare unrelated rows (and crash on a size mismatch,
        which is how this was found).
        """
        if not model.training or (self.gate_lambda == 0 and self.gate_floor_lambda == 0
                                  and self.gate_ortho_lambda == 0):
            return total
        from .moe2 import (cond_moe_blocks, condition_from_paths, expert_floor_loss,
                           gate_supervision_loss)

        blocks = cond_moe_blocks(model)
        if not blocks:
            return total
        paths = batch.get("im_file") if isinstance(batch, dict) else None
        if not paths:
            return total
        targets = condition_from_paths(paths)
        # Belt and braces: if the stashed logits do not line up with this batch,
        # they are stale and must not be used.
        stale = [b for b in blocks if b.last_logits is not None
                 and b.last_logits.shape[0] != targets.shape[0]]
        if stale:
            return total
        # Each term is scaled by its OWN lambda and summed, rather than folded
        # into one number and multiplied by gate_lambda. The earlier shape made
        # a floor-only run (gate_lambda = 0) silently contribute nothing.
        term = 0.0
        if self.gate_lambda:
            bce = gate_supervision_loss(model, targets, pos_weight=self.gate_pos_weight)
            if self.gate_count_lambda:
                from .moe2 import routing_cost

                cost = routing_cost(model, targets, weight_count=self.gate_count_lambda)
                if isinstance(cost, torch.Tensor):
                    self._last_count_loss = float(cost.detach())
                    bce = bce + cost
            if isinstance(bce, torch.Tensor):
                self._last_gate_loss = float(bce.detach())
                term = term + self.gate_lambda * bce
        if self.gate_ortho_lambda:
            from .moe2 import expert_orthogonality_loss

            o = expert_orthogonality_loss(model)
            if isinstance(o, torch.Tensor):
                self._last_ortho = float(o.detach())
                term = term + self.gate_ortho_lambda * o
        if self.gate_floor_lambda:
            floor = expert_floor_loss(model, tau=self.gate_floor_tau)
            if isinstance(floor, torch.Tensor):
                self._last_floor_loss = float(floor.detach())
                term = term + self.gate_floor_lambda * floor
        if isinstance(term, torch.Tensor):
            if isinstance(total, torch.Tensor) and total.ndim > 0:
                total = total.clone()
                total[0] = total[0] + term
            else:
                total = total + term
        return total

    def build_dataset(self, img_path, mode="train", batch=None):
        """Training set carries each degraded image's clear twin when asked to.

        Train only. The validator has no use for the extra key and would have to
        be taught to ignore it.
        """
        ds = super().build_dataset(img_path, mode, batch)
        if self.restore_lambda == 0 or mode != "train":
            return ds
        from .paired import PairedYOLODataset

        if getattr(self.args, "mosaic", 0.0):
            LOGGER.warning("restore_lambda is set but mosaic is ON: the twin cannot be "
                           "given the same augmentation, so the pairing is invalid")
        # Reassigning __class__ keeps the already-scanned labels and cache, but
        # skips __init__, so the twin index is attached explicitly.
        from .paired import attach_twins

        ds.__class__ = PairedYOLODataset
        n = attach_twins(ds)
        LOGGER.info(f"paired restoration: {n}/{ds.n_sampled} degraded samples have a "
                    f"clear twin ({ds.ni - ds.n_sampled} twins appended as targets)")
        return ds

    def _add_restoration(self, model, total, batch):
        """SmoothL1 between the block output on a degraded image and on its twin.

        Runs LAST in the loss wrapper: the twin forward pass overwrites the
        block's stashed routing state, so gate supervision and utilisation
        logging have to have finished first.
        """
        if self.restore_lambda == 0 or not model.training:
            return total
        from .moe2 import cond_moe_blocks
        from .paired import condition_token, restoration_loss

        blocks = cond_moe_blocks(model)
        twin = batch.get("twin_img") if isinstance(batch, dict) else None
        if not blocks or twin is None or blocks[0].last_out is None:
            return total
        blk = blocks[0]
        student = blk.last_out
        paths = batch.get("im_file") or []
        degraded = torch.tensor([condition_token(f) in ("fog", "night") for f in paths],
                                device=student.device)
        if degraded.shape[0] != student.shape[0] or not degraded.any():
            return total

        with torch.no_grad():
            model(twin.to(student.device, non_blocking=True).float() / 255)
            target = blk.last_out
        if target is None or target.shape != student.shape:
            return total

        loss = restoration_loss(student, target, degraded, beta=self.restore_beta)
        if isinstance(loss, torch.Tensor):
            self._last_restore = float(loss.detach())
            term = self.restore_lambda * loss
            if isinstance(total, torch.Tensor) and total.ndim > 0:
                total = total.clone()
                total[0] = total[0] + term
            else:
                total = total + term
        return total

    def _finish(self, model, total, items, use_aux: bool):
        """Add the routing auxiliary term and record utilisation.

        `_accumulate` runs unconditionally: a condition-gated run has no legacy
        auxiliary loss, but its routing statistics are the whole point of the
        experiment and must still be recorded.
        """
        if not use_aux:
            self._accumulate(model)
            return total, items
        aux = routing_aux_loss(model, mode=self.moe_aux)
        if isinstance(aux, torch.Tensor):
            self._last_aux = float(aux.detach())
            term = self.moe_lambda * aux
            # The loop does `self.loss = loss.sum()`, so a scalar added to a
            # vector-valued loss would be counted once per component. Add it to a
            # single element instead, leaving `items` (what gets displayed as
            # box/cls/dfl) untouched.
            if isinstance(total, torch.Tensor) and total.ndim > 0:
                total = total.clone()
                total[0] = total[0] + term
            else:
                total = total + term
        self._accumulate(model)
        return total, items

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
        from .moe2 import routing_report

        # CLEAN activation rates -- never the noisy mask. Logging the noisy one
        # is exactly what hid the previous router collapse. Training-mode only,
        # for the same staleness reason as the gate supervision.
        if model.training:
            self._cond_report = routing_report(model)
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
        out["moe/gate_loss"] = getattr(self, "_last_gate_loss", 0.0)
        out["moe/count_loss"] = getattr(self, "_last_count_loss", 0.0)
        out["moe/floor_loss"] = getattr(self, "_last_floor_loss", 0.0)
        out["moe/restore_loss"] = getattr(self, "_last_restore", 0.0)
        out["moe/ortho_loss"] = getattr(self, "_last_ortho", 0.0)
        out.update(self._cond_report)
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
