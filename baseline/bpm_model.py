import pytorch_lightning as pl
from sklearn.metrics import confusion_matrix, f1_score
import torch
import torch.nn as nn
from torch.nn.functional import relu, sigmoid, softmax
from transformers import BertModel

import argparse
import os 

from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from torch.utils.data import DataLoader
from bpm_utils import BCDataset


LSTM_DIM = 100 #TODO: find the appropriate size


class BPM_MT(pl.LightningModule):
    def __init__(self, lm_name="bert-base-uncased"):
        super(BPM_MT, self).__init__()

        # Model Architecture
        # Transformer
        self.lm = BertModel.from_pretrained(
            lm_name,
            attention_probs_dropout_prob=0.3,
            hidden_dropout_prob=0.3
        )
        # Additional Parameter
        self.lstm = nn.LSTM(
            13, LSTM_DIM, bidirectional=True, batch_first=True
        )

        self.fc1 = nn.Linear(768 + LSTM_DIM, 128)
        self.fc2 = nn.Linear(128, 3)
        self.fc3 = nn.Linear(768, 64)
        self.fc4 = nn.Linear(64, 3)

        self.label_to_idx = {
            "NO-BC": 0,
            "CONTINUER": 1,
            "ASSESSMENT": 2
        }

        # this property activates manual optimization.
        self.automatic_optimization = False

    def configure_optimizers(self):
        """Define algorithms and schedulers used in optimization."""
        # optimization uses SGD for transformer params and Adam for other params
        transf_opt = torch.optim.SGD(self.lm.parameters(), lr=0.0005)
        other_opt = torch.optim.Adam([
            *self.lstm.parameters(),
            *self.fc1.parameters(),
            *self.fc2.parameters(),
            *self.fc3.parameters(),
            *self.fc4.parameters()
        ], lr=0.0005)

        return transf_opt, other_opt

    def forward(
        self,
        text_input_ids,
        text_attention_mask,
        acoustic_input,
        train=False
    ):
        """Define computation performed at model call."""
        lm_output = self.lm(
            input_ids=text_input_ids, attention_mask=text_attention_mask
        )[0]

        # encoded vector of the CLS token is used as the lexical representation
        cls_token_encoding = lm_output[:,0,:]

        ## Main Task

        # feed mfcc feature vectors into LSTM layers for acoustic representation
        outputs, ((forward, backward), final_cell) = self.lstm(acoustic_input)
        # add the final hidden states from the forward and backward pass
        acoustic_encoding = torch.add(forward, backward)

        # acoustic and lexical hidden representations are concatenated
        combined_encoding = torch.cat((acoustic_encoding, cls_token_encoding), 1)

        # feed into fully connected FC128 layer with final softmax normalisation
        # outputing a probability distribution across the 4 BC categories
        hidden_emb = relu(self.fc1(combined_encoding))
        main_output = torch.softmax(self.fc2(hidden_emb), 1)

        ## Sub-Task

        sub_output = None
        if train:
            hidden_emb = relu(self.fc3(cls_token_encoding))
            # predicted normalized match count per sentiment category
            sub_output = torch.sigmoid(self.fc4(hidden_emb))

        return main_output, sub_output

    def training_step(self, batch, batch_idx):
        """Compute and return training loss."""
        main_criterion = nn.CrossEntropyLoss()
        sub_criterion = nn.BCELoss(reduction='mean')
        
        # compute loss
        main_output, sub_output = self.forward(
            batch["input_ids"], batch["attention_mask"],
            batch["acoustic_input"], train=True
        )
        main_labels = torch.tensor(
            [self.label_to_idx[l] for l in batch["main_labels"]],
            device=main_output.device
        )
        main_loss, sub_loss = (
            main_criterion(main_output, main_labels)
            , sub_criterion(sub_output, batch["sub_labels"])
        )
        total_loss = (1-0.1)*main_loss + 0.1*sub_loss

        transf_opt, other_opt = self.optimizers()
        
        # optimize through back pass
        transf_opt.zero_grad()
        other_opt.zero_grad()
        self.manual_backward(total_loss)
        transf_opt.step()
        other_opt.step()

        # record loss values
        self.log("main_loss", main_loss, on_step=False, on_epoch=True)
        self.log("sub_loss", sub_loss, on_step=False, on_epoch=True)
        self.log("total_loss", total_loss, progbar=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        """Compute and return validation loss."""
        main_criterion = nn.CrossEntropyLoss()

        # compute loss
        main_output, _ = self.forward(
            batch["input_ids"], batch["attention_mask"],
            batch["acoustic_input"], train=False
        )
        main_labels = torch.tensor(
            [self.label_to_idx[l] for l in batch["main_labels"]],
            device=main_output.device
        )
        main_loss = main_criterion(main_output, main_labels)
        self.log("val_loss", main_loss, on_step=False, on_epoch=True)
        
        return main_labels, torch.argmax(main_output, dim=1)

    def validation_epoch_end(self, validation_step_out):
        """Call at validation time in the very end of the epoch."""
        gold = []
        pred = []

        for g, p in validation_step_out:
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


if __name__ == "__main__":
    pl.seed_everything(1)

    # configure commandline interface
    parser = argparse.ArgumentParser(
        description="Interface to train the BPM_MT (2021) Model."
    )
    parser.add_argument(
        "--load", metavar="CKPT_PATH", default=None,
        help="path to existing model state"
    )
    parser.add_argument(
        "--save", metavar="CKPT_NAME", default="checkpoint",
        help="name to save model state under"
    )

    args = parser.parse_args()
    print(__file__)
    path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    print(path)

    # prepare data
    print("Load data ...")
    train_dataset = BCDataset(
        os.path.join(path, "data", "swb", "conversations.train"),
        os.path.join(
            path, "data", "swb",
            "utterance_is_backchannel_with_sentiment_with_context.csv"
        ),
        os.path.join(path, "data", "swb", "swb_audios")
    )
    train_data = DataLoader(train_dataset, batch_size=16)
    valid_dataset = BCDataset(
        os.path.join(path, "data", "swb", "conversations.valid"),
        os.path.join(
            path, "data", "swb",
            "utterance_is_backchannel_with_sentiment_with_context.csv"
        ),
        os.path.join(path, "data", "swb", "swb_audios")
    )
    valid_data = DataLoader(valid_dataset, batch_size=16)

    # prepare model
    print("Build model ...")
    model = BPM_MT("bert-base-uncased")
    print("COMPLETE")

    # configure training procedure
    callbacks = [
        ModelCheckpoint(
            monitor="f1",
            mode="max",
            dirpath=os.path.join(
                path, "data", "model_checkpoints"
            ),
            filename=args.save + "-{epoch}-{step}"
        ),
        EarlyStopping(
            monitor="f1",
            mode="max",
            patience=5,
            strict=True,
            verbose=False,
        ),
        LearningRateMonitor()
    ]
    trainer = pl.Trainer(
        callbacks=callbacks,
        deterministic=True,
        max_epochs=60,
        accelerator="gpu"
    )

    # actual training
    trainer.fit(
        model,
        ckpt_path=args.load,
        train_dataloaders=train_data,
        val_dataloaders=valid_data
    )
