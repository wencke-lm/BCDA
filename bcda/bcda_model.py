"""bcda_model.py - Supervised backchannel prediction model"""
import pytorch_lightning as pl
from sklearn.metrics import confusion_matrix, f1_score
import torch
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup

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

        feat_n = 0
        # audio feature processing
        if self.confg["use_audio"]:
            self.encoder = Encoder(**encoder_confg)
            self.transformer = Transformer(**predictor_confg)
            feat_n += predictor_confg["dim"]
        # text feature processing
        if self.confg["use_text"]:
            self.text_encoder = SpartaModel(
                self.confg["bert_dropout"],
                self.confg["hist_n"]
            )
            feat_n += 768
        # add dialogue act classification objective
        if self.confg["use_multitask"]:
            self.sub_classification_head = nn.Linear(
                768, 41
            )
            self.sub_label_to_idx = {
                "": -1,
                "%":0,
                "^2":1,
                "^g":2,
                "^h":3,
                "^q":4,
                "aa":5,
                "ad":6,
                "am":7,
                "ar":8,
                "b":9,
                "b^m":10,
                "ba":11,
                "bc":12,
                "bd":13,
                "bf":14,
                "bh":15,
                "bk":16,
                "br":17,
                "cc":18,
                "fa":19,
                "fc":20,
                "fp":21,
                "ft":22,
                "h":23,
                "na":24,
                "nd":25,
                "ng":26,
                "nn":27,
                "no":28,
                "ny":29,
                "qh":30,
                "qo":31,
                "qrr":32,
                "qw":33,
                "qw^d":34,
                "qy":35,
                "qy^d":36,
                "sd":37,
                "sv":38,
                "t1":39,
                "t3":40,
            }

        self.classification_head = nn.Linear(
            feat_n, 3
        )
        self.label_to_idx = {
            "NO-BC": 0,
            "CONTINUER": 1,
            "ASSESSMENT": 2
        }

        for n, p in self.named_parameters():
            print(n, p.requires_grad)

        self.save_hyperparameters()
        # this property activates manual optimization.
        self.automatic_optimization = False

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
        transf_opt = torch.optim.AdamW([
                *(self.encoder.parameters() if self.confg["use_audio"] else []),
                *(self.transformer.parameters() if self.confg["use_audio"] else []),
                *(self.text_encoder.model.parameters() if self.confg["use_text"] else [])
            ],
            lr=0.2*self.confg["optimizer"]["learning_rate"],
            betas=self.confg["optimizer"]["betas"],
            weight_decay=self.confg["optimizer"]["weight_decay"]
        )
        other_opt = torch.optim.AdamW([
                *self.classification_head.parameters(),
                *(self.sub_classification_head.parameters() if self.confg["use_multitask"] else []),
                *(self.text_encoder.ta_attention.parameters() if self.confg["use_text"] else [])
            ],
            lr=self.confg["optimizer"]["learning_rate"],
            betas=self.confg["optimizer"]["betas"],
            weight_decay=self.confg["optimizer"]["weight_decay"]
        )

        it_per_epoch = 6281
        freq = self.confg["optimizer"]["lr_scheduler_freq"]

        transf_lr = get_linear_schedule_with_warmup(
            optimizer=transf_opt,
            num_warmup_steps=1*it_per_epoch/freq,
            num_training_steps=10*it_per_epoch/freq
        )
        other_lr = torch.optim.lr_scheduler.LinearLR(
            optimizer=other_opt,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=10*it_per_epoch/freq
        )

        return [transf_opt, other_opt], [transf_lr, other_lr]

    def forward(self, x):
        """Define computation performed at model call."""

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
            feat_emb = torch.cat((audio_emb, text_emb), 1)

        if not self.confg["use_multitask"]:
            return self.classification_head(feat_emb), None
        return (
            self.classification_head(feat_emb),
            self.sub_classification_head(text_emb)
        )

    def _shared_step(self, batch, batch_idx):
        """Prepare evaluation & backward pass."""
        out, sub_out = self(batch)

        labels = torch.tensor(
            [self.label_to_idx[l] for l in batch["labels"]],
            device=out.device
        )
        weights = torch.tensor([0.80, 1.3, 1.1], device=out.device)
        loss = nn.functional.cross_entropy(
            out, labels, weight=weights
        )

        if self.confg["use_multitask"]:
            sub_labels = torch.tensor(
                [self.sub_label_to_idx[l] for l in batch["sub_labels"]],
                device=out.device
            )
            sub_loss = nn.functional.cross_entropy(
                sub_out[sub_labels != -1], sub_labels[sub_labels != -1]
            )
            if not sub_loss.isnan():
                loss = 0.7*loss + 0.3*sub_loss
                self.log("sub_loss", sub_loss, on_epoch=True)

        return loss, labels, out

    def training_step(self, batch, batch_idx):
        """Compute and return training loss."""
        loss, _, _ = self._shared_step(batch, batch_idx)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log("loss", loss, prog_bar=True)

        transf_opt, other_opt = self.optimizers()
        transf_lr, other_lr = self.lr_schedulers()

        # optimize through back pass
        transf_opt.zero_grad()
        other_opt.zero_grad()
        self.manual_backward(loss)
        transf_opt.step()
        other_opt.step()

        if (batch_idx + 1) % self.confg["optimizer"]["lr_scheduler_freq"] == 0:
            transf_lr.step()
            other_lr.step()

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
