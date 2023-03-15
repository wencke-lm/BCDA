"""vap_model.py - Self-supervisedly trained voice activity projection model"""
from collections import Counter

import pytorch_lightning as pl
from sklearn.metrics import precision_recall_fscore_support
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

    def configure_optimizers(self):
        """Define algorithms and schedulers used in optimization."""
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

    def forward(self, x):
        """Define computation performed at model call."""
        return self.predictor(self.encoder(x))

    def _shared_step(self, batch, batch_idx):
        """Prepare evaluation & backward pass."""
        labels = self.predictor.classification_head.extract_gold_labels(
            batch["va"]
        ).flatten()

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
        """Compute and return training loss."""
        loss, _, _ = self._shared_step(batch, batch_idx)
        self.log("train_loss", loss, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Compute and return validation loss."""
        loss, gold, pred = self._shared_step(batch, batch_idx)
        self.log("val_loss", loss, on_step=False, on_epoch=True)

        return loss, pred[:, -1], *batch["event"]

    def on_train_epoch_start(self):
        """Call at training time in the very beginning of the epoch."""
        if self.current_epoch == self.confg["optimizer"]["train_encoder_epoch"]:
            self.encoder.set_cpc_mode(freeze=False)

    def validation_epoch_end(self, validation_step_out):
        """Call at validation time in the very end of the epoch."""
        true_events = []
        pred_events = []

        for _, pred_batch, event_batch, speaker_batch in validation_step_out:
            for pred, event, speaker in zip(
                pred_batch, event_batch, speaker_batch
            ):
                pred_next_speaker = (
                    self.predictor.classification_head.get_next_speaker(
                        torch.nn.functional.softmax(pred, dim=-1)
                    )
                )

                if speaker == pred_next_speaker:
                    pred_events.append("HOLD")
                else:
                    pred_events.append("SHIFT")
                true_events.append(event)

        prec, recall, (hold_f1, shift_f1), _ = precision_recall_fscore_support(
            true_events, pred_events, labels=["HOLD", "SHIFT"], average=None
        )

        print("Precision:", prec)
        print("Recall:", recall)

        weights = Counter(true_events)
        total_f1 = (
            (weights["HOLD"]*hold_f1 + weights["SHIFT"]*shift_f1)/
            sum(weights.values())
        )

        self.log("hold_f1", hold_f1)
        self.log("shift_f1", shift_f1)
        self.log("total_f1", total_f1)
