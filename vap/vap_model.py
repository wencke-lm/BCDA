"""vap_model.py - Self-supervisedly trained voice activity projection model"""
import pytorch_lightning as pl
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
        # accuracy and confusion matrix
        all_pred = dict()
        true_pred = dict()

        true_hold = 0
        false_hold = 0
        true_shift = 0
        false_shift = 0

        for _, gold, pred, batch in validation_step_outputs:
            for g, p in zip(gold.tolist(), pred.argmax(dim=-1).tolist()):
                all_pred[p] = all_pred.get(p, 0) + 1
                if g == p:
                    true_pred[p] = true_pred.get(p, 0) + 1

            for i in range(pred.shape[0]):
                sample = {k: v[i] for k, v in batch.items()}
                event = is_shift_or_hold(sample, 5, 100)
                if event is not None:
                    true_next_speaker = event[1]
                    pred_next_speaker = self.predictor.classification_head.get_next_speaker(pred[i])
                    if event[0] == "HOLD":
                        if true_next_speaker == pred_next_speaker:
                            true_hold += 1
                        else:
                            false_hold += 1
                    if event[0] == "SHIFT":
                        if true_next_speaker == pred_next_speaker:
                            true_shift += 1
                        else:
                            false_shift += 1

        print("\n")
        for label in all_pred:
            print(label, all_pred[label], true_pred.get(label), sep="\t")
        print("\n")

        accuracy = sum(true_pred.values()) / sum(all_pred.values())
        self.log("accuracy_epoch", accuracy)

        if true_shift:
            shift_recall = true_shift / (true_shift + false_hold)
            shift_prec = true_shift / (true_shift + false_shift)
            shift_f1 = (2*shift_recall*shift_prec)/(shift_recall+shift_prec)
        else:
            shift_f1 = 0 
        if true_hold:
            hold_recall = true_hold / (true_hold + false_shift)
            hold_prec = true_hold / (true_hold + false_hold)
            hold_f1 = (2*hold_recall*hold_prec)/(hold_recall+hold_prec)
        else:
            hold_f1 = 0

        try:
            total_f1 = ((true_shift + false_hold)*shift_f1 + (true_hold + false_shift)*hold_f1)/(true_shift + false_hold + true_hold + false_shift)
        except:
            total_f1 = 0

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
