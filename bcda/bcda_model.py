"""bcda_model.py - Supervised backchannel prediction model"""
import pytorch_lightning as pl
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn

from vap.encoder import Encoder
from vap.predictor import Transformer
from vap.vap_model import VAPModel


class BCDAModel(pl.LightningModule):
    def __init__(self, confg, encoder_confg, predictor_confg):
        super().__init__()

        self.confg = confg
        self.encoder = Encoder(**encoder_confg)
        self.transformer = Transformer(**predictor_confg)

        self.classification_head = nn.Linear(
            predictor_confg["dim"], 3
        )
        self.label_to_idx = {
            "NO-BC": 0,
            "CONTINUER": 1,
            "ASSESSMENT": 2
        }

        self.save_hyperparameters()

    def load_pretrained_vap(self, model_state):
        """Load self-supervised Voice Activity Projection model."""
        pretrained_model = VAPModel.load_from_checkpoint(model_state)
        pretrained_model_dict = pretrained_model.state_dict()

        # update state dict to reflect missing projection head
        for k in list(pretrained_model_dict.keys()):
            if k.startswith("predictor.classification_head."):
                pretrained_model_dict.pop(k)
            elif k.startswith("predictor."):
                v = pretrained_model_dict.pop(k)
                pretrained_model_dict[k.replace("predictor.", "")] = v

        self.load_state_dict(pretrained_model_dict, strict=False)

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

    def forward(self, x, mask=None):
        """Define computation performed at model call."""
        return self.classification_head(
            self.transformer(self.encoder(x), mask=mask)[:, -1]
        )

    def _shared_step(self, batch, batch_idx):
        """Prepare evaluation & backward pass."""
        out = self(batch, mask=batch["masks"])
        labels = torch.tensor(
            [self.label_to_idx[l] for l in batch["labels"]],
            device=out.device
        )

        loss = nn.functional.cross_entropy(
            out, labels
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

        return loss, gold, torch.argmax(pred, dim=1)

    def validation_epoch_end(self, validation_step_out):
        """Call at validation time in the very end of the epoch."""
        gold = []
        pred = []

        for _, g, p in validation_step_out:
            gold.extend(g.tolist())
            pred.extend(p.tolist())

        conf_mtrx = confusion_matrix(
            # ["NO-BC", "CONTINUER", "ASSESSMENT"]
            gold, pred, labels= [0, 1, 2]
        )
        print(conf_mtrx)

        return conf_mtrx
