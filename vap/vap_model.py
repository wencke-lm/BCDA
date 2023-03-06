"""vap_model.py - Self-supervisedly trained voice activity projection model"""
from collections import Counter

import pytorch_lightning as pl
from sklearn.metrics import f1_score
import torch
import torch.nn as nn

from vap.encoder import Encoder
from vap.events import is_shift_or_hold
from vap.predictor import Predictor


class VAPModel(pl.LightningModule):
    """Voice Activity Projection Model.

    Based on ...

    The model processes acoustic input features made up
    of a combined speech waveform and the activity status of
    each speaker at time point t, together with a longer
    history of their relative participation so far.

    For more details on training methods, see:
    https://pytorch-lightning.readthedocs.io/en/
    stable/common/lightning_module.html

    """
    def __init__(self, confg, encoder_confg, predictor_confg):
        super().__init__()

        self.confg = confg
        self.encoder = Encoder(**encoder_confg)
        self.predictor = Predictor(**predictor_confg)

        if self.confg["class_weight"]:
            n_classes = self.predictor.classification_head.n_classes
            self.class_dist = torch.ones(
                (n_classes, ),
                device=encoder_confg["device"],
                requires_grad=False
            )

        self.save_hyperparameters()

    def forward(self, x):
        return self.predictor(self.encoder(x))

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._shared_step(batch, batch_idx)
        self.log("train_loss", loss, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, gold, pred = self._shared_step(batch, batch_idx)
        self.log("val_loss", loss, on_step=False, on_epoch=True)

        return loss, gold, pred, batch

    def validation_epoch_end(self, validation_step_outputs):
        all_pred = dict()
        true_pred = dict()

        true_events = []
        pred_events = []

        for _, gold, pred, batch in validation_step_outputs:
            for g, p in zip(gold.tolist(), pred.argmax(dim=-1).tolist()):
                all_pred[p] = all_pred.get(p, 0) + 1
                if g == p:
                    true_pred[p] = true_pred.get(p, 0) + 1

            for i in range(pred.shape[0]):
                sample = {k: v[i] for k, v in batch.items()}
                event, curr_speaker = is_shift_or_hold(sample, 5, 100)
                if event is not None:
                    pred_next_speaker = (
                        self.predictor.classification_head
                        .get_next_speaker(pred[i])
                    )

                    true_events.append(event)
                    if curr_speaker == pred_next_speaker:
                        pred_events.append("HOLD")
                    else:
                        pred_events.append("SHIFT")

        weights = Counter(true_events)
        hold_f1, shift_f1 = f1_score(
            true_events, pred_events, labels=["HOLD", "SHIFT"], average=None
        )
        total_f1 = (
            weights["HOLD"]*hold_f1 + weights["SHIFT"]*shift_f1/
            sum(weights.values())
        )

        self.log("hold_f1", hold_f1)
        self.log("shift_f1", shift_f1)
        self.log("total_f1", total_f1)

    def _shared_step(self, batch, batch_idx):
        labels = self.predictor.classification_head.get_gold_label(
            batch["labels"]
        )
        out = self(batch)

        if self.confg["class_weight"]:
            mw = self.confg["min_weight"]
            s = ((1-mw)/self.class_dist.max())*(self.class_dist)
            loss = nn.functional.cross_entropy(
                out, labels, weight=(1-s)
            )
            if self.current_epoch == 0:
                for l in labels.tolist():
                    self.class_dist[l] += 1
        else:
            loss = nn.functional.cross_entropy(
                out, labels
            )

        return loss, labels, out

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.confg["optimizer"]["learning_rate"],
            betas=self.confg["optimizer"]["betas"],
            weight_decay=self.confg["optimizer"]["weight_decay"]
        )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=opt,
            T_max=self.confg["optimizer"]["lr_scheduler_tmax"]
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": self.confg["optimizer"]["lr_scheduler_interval"],
                "frequency": self.confg["optimizer"]["lr_scheduler_freq"]
            },
        }

    def on_train_epoch_start(self):
        if self.current_epoch == self.confg["optimizer"]["train_encoder_epoch"]:
            self.encoder.set_cpc_mode(freeze=False)
