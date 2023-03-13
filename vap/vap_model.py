"""vap_model.py - Self-supervisedly trained voice activity projection model"""
from collections import Counter

import pytorch_lightning as pl
from sklearn.metrics import f1_score
import torch
import torch.nn as nn

from vap.encoder import Encoder
from vap.predictor import Predictor


class VAPModel(pl.LightningModule):
    """Voice Activity Projection Model.

    Based on https://arxiv.org/pdf/2205.09812.pdf

    The model processes acoustic input features made up
    of a combined speech waveform and the activity status of
    each speaker at time point t, together with a longer
    history of their relative participation so far.

    For more details on class methods, see:
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

    def _shared_step(self, batch, batch_idx):
        labels = []
        for b in batch["va"]:
            labels.append(
                self.predictor.classification_head.extract_gold_labels(
                    b
                )
            )
        labels = torch.stack(labels).flatten()

        # split predictive window from model input window
        va_pred_strides = int(self.predictor.classification_head.pred_bins[0])
        wave_pred_frames = int(
            va_pred_strides*(batch["waveform"].shape[-1]/batch["va"].shape[-1])
        )

        batch["va"] = batch["va"][:, :, :-va_pred_strides]
        batch["va_hist"] = batch["va_hist"][:, :, :-va_pred_strides]
        batch["waveform"] = batch["waveform"][:, :, :-wave_pred_frames]

        out = self(batch)

        if self.confg["class_weight"]:
            mw = self.confg["min_weight"]
            s = ((1-mw)/self.class_dist.max())*(self.class_dist)
            loss = nn.functional.cross_entropy(
                out.flatten(start_dim=0, end_dim=1), labels, weight=(1-s)
            )
            if self.current_epoch == 0:
                for l in labels.tolist():
                    self.class_dist[l] += 1
        else:
            loss = nn.functional.cross_entropy(
                out.flatten(start_dim=0, end_dim=1), labels
            )

        return loss, labels, out

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._shared_step(batch, batch_idx)
        self.log("train_loss", loss, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, gold, pred = self._shared_step(batch, batch_idx)
        self.log("val_loss", loss, on_step=False, on_epoch=True)

        return loss, gold, pred[:, 0], batch["event"][0], batch["event"][1]

    def on_train_epoch_start(self):
        if self.current_epoch == self.confg["optimizer"]["train_encoder_epoch"]:
            self.encoder.set_cpc_mode(freeze=False)

    def validation_epoch_end(self, validation_step_outputs):
        all_pred = {}
        true_pred = {}

        true_events = []
        pred_events = []

        for _, gold, pred, event, pre_speaker in validation_step_outputs:
            for g, p in zip(gold.tolist(), pred.argmax(dim=-1).tolist()):
                all_pred[p] = all_pred.get(p, 0) + 1
                if g == p:
                    true_pred[p] = true_pred.get(p, 0) + 1

            for i in range(pred.shape[0]):
                e, sp = event[i], pre_speaker[i]

                pred_next_speaker = (
                    self.predictor.classification_head
                    .get_next_speaker(pred[i])
                )

                true_events.append(e)
                if sp == pred_next_speaker:
                    pred_events.append("HOLD")
                else:
                    pred_events.append("SHIFT")

        print("\n")
        for label in all_pred:
            print(label, all_pred[label], true_pred.get(label), sep="\t")
        print("\n")

        weights = Counter(true_events)
        hold_f1, shift_f1 = f1_score(
            true_events, pred_events, labels=["HOLD", "SHIFT"], average=None
        )
        total_f1 = (
            (weights["HOLD"]*hold_f1 + weights["SHIFT"]*shift_f1)/
            sum(weights.values())
        )

        print(hold_f1, shift_f1, total_f1)

        self.log("hold_f1", hold_f1)
        self.log("shift_f1", shift_f1)
        self.log("total_f1", total_f1)
