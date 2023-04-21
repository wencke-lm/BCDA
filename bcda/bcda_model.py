"""bcda_model.py - Supervised backchannel prediction model"""
import pytorch_lightning as pl
from sklearn.metrics import confusion_matrix, f1_score
import torch
import torch.nn as nn

from sparta.sparta import SpartaModel
from vap.encoder import Encoder
from vap.predictor import Transformer
from vap.vap_model import VAPModel


class BCDAModel(pl.LightningModule):
    def __init__(
        self,
        confg,
        encoder_confg,
        predictor_confg
    ):
        super().__init__()

        self.confg = confg

        self.feat_emb_dim = predictor_confg["dim"]
        # audio feature processing
        if self.confg["use_audio"]:
            self.encoder = Encoder(**encoder_confg)
            self.transformer = Transformer(**predictor_confg)
        # text feature processing
        if self.confg["use_text"]:
            self.text_encoder = SpartaModel(
                self.feat_emb_dim,
                self.confg["bert_dropout"],
                self.confg["hist_n"]
            )

        self.classification_head = nn.Linear(
            predictor_confg["dim"], 3
        )
        self.label_to_idx = {
            "NO-BC": 0,
            "CONTINUER": 1,
            "ASSESSMENT": 2
        }

        for n, p in self.named_parameters():
            print(n, p.requires_grad)

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

        for p in self.encoder.parameters():
            p.requires_grad_(False)
        for p in self.transformer.parameters():
            p.requires_grad_(False)

    def configure_optimizers(self):
        """Define algorithms and schedulers used in optimization."""
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.confg["optimizer"]["learning_rate"],
            betas=self.confg["optimizer"]["betas"],
            weight_decay=self.confg["optimizer"]["weight_decay"]
        )
        # if self.confg["optimizer"]["train_transformer_epoch"] != -1:
        #    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        #        optimizer=opt,
        #        milestones=[self.confg["optimizer"]["train_transformer_epoch"]],
        #        gamma=0.1
        #    )
        #else:
        #    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #        optimizer=opt,
        #        T_max=self.confg["optimizer"]["lr_scheduler_tmax"]
        #    )
        warm_up = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, end_factor=1, total_iters=6280)
        cool_down = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1.0, end_factor=0.1, total_iters=307720)
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(opt, schedulers=[warm_up, cool_down], milestones=[6280])
        
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step", # self.confg["optimizer"]["lr_scheduler_interval"],
                "frequency": 1 # self.confg["optimizer"]["lr_scheduler_freq"]
            },
        }

    def forward(self, x):
        """Define computation performed at model call."""
        # add batch dimension if missing

        if self.confg["use_audio"]:
            audio_emb = self.transformer(
                self.encoder(x),
                mask=x["masks"]
            )[:, -1]

            if not self.confg["use_text"]:
                feat_emb = audio_emb

        if self.confg["use_text"]:
            text_emb = self.text_encoder(
                x["text_input_ids"],
                x["text_attention_mask"]
            )
            if not self.confg["use_audio"]:
                feat_emb = text_emb

        if self.confg["use_audio"] and self.confg["use_text"]:
            feat_emb = torch.maximum(audio_emb, text_emb)

        return self.classification_head(feat_emb)

    def _shared_step(self, batch, batch_idx):
        """Prepare evaluation & backward pass."""
        out = self(batch)
        labels = torch.tensor(
            [self.label_to_idx[l] for l in batch["labels"]],
            device=out.device
        )

        loss = nn.functional.cross_entropy(
            out, labels, weight=torch.tensor([0.85, 1.125, 1.025], device=out.device)
        )

        return loss, labels, out

    def training_step(self, batch, batch_idx):
        """Compute and return training loss."""
        loss, _, _ = self._shared_step(batch, batch_idx)
        self.log("train_loss", loss, on_step=False, on_epoch=True)

        return loss

    def on_train_epoch_start(self):
        """Call at train time at the very start of the epoch."""
        if self.current_epoch == self.confg["optimizer"]["train_transformer_epoch"]:
            for name, p in self.named_parameters():
                if name.startswith("transformer") or name.startswith("encoder.va"):
                    p.requires_grad_(True)

        if self.current_epoch == self.confg["optimizer"]["train_llm_epoch"]:
            for name, p in self.named_parameters():
                if name.startswith("text_encoder.model"):
                    p.requires_grad_(True)
                    print(name, p.requires_grad)

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
            gold, pred, labels=[0, 1, 2]
        )
        print(conf_mtrx)

        f1 = f1_score(gold, pred, average="weighted")
        self.log("f1", f1)

        return conf_mtrx
