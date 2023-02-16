"""vap_model.py - Self-supervisedly trained voice activity projection model"""
import pytorch_lightning as pl
import torch
import torch.nn as nn

from vap.encoder import Encoder
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

        self.save_hyperparameters()

    def forward(self, x):
        return self.predictor(self.encoder(x))

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx)
        self.log("train_los", loss)

        return loss

    # def validation_step(self, batch, batch_idx):
        # loss = self._shared_step(batch, batch_idx)
        # self.log("val_loss", loss, on_epoch=True)

        # return loss

    def _shared_step(self, batch, batch_idx):
        pred_idx = self.confg["pred_window"]*self.confg["n_stride"]
        pred_idx2 = self.confg["pred_window"]*self.confg["sample_rate"]

        inpt = {k:v[:, :, :-pred_idx] if k in {"va", "va_hist"} else v[:, :, :-pred_idx2] for k, v in batch.items()}

        labels = self.predictor.classification_head.get_gold_label(
            batch["va"][:, :, -pred_idx:]
        )
        out = self(inpt)

        loss = nn.functional.cross_entropy(out, labels)

        return loss

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
