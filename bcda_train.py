"""test.py - Interface to test voice activity projection model"""
import argparse
import os
import sys

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from torch.utils.data import DataLoader
import yaml

from vap.data_loader import SwitchboardCorpus
from bcda.bcda_model import BCDAModel
from bcda.utils import BCDataset


BC_DATA = os.path.join("data", "swb", "utterance_is_backchannel.csv")


def train(cfg_dict, init_state, ckpt_load, ckpt_save):
    pl.seed_everything(cfg_dict["seed"])

    callbacks = [
        ModelCheckpoint(
            mode=cfg_dict["training"]["checkpoint"]["mode"],
            monitor=cfg_dict["training"]["checkpoint"]["monitor"],
            dirpath=cfg_dict["training"]["checkpoint"]["dirpath"],
            filename=ckpt_save + "-{epoch}-{step}"
        ),
        EarlyStopping(
            monitor=cfg_dict["training"]["early_stopping"]["monitor"],
            mode=cfg_dict["training"]["early_stopping"]["mode"],
            patience=cfg_dict["training"]["early_stopping"]["patience"],
            strict=True,
            verbose=False,
        ),
        LearningRateMonitor()
    ]

    # prepare data
    print("Load data ...")
    split_info = cfg_dict["data"].pop("split")

    train_swb = SwitchboardCorpus(
        split_info=split_info["train_split"], **cfg_dict["data"]
    )
    train_dataset = BCDataset(train_swb, BC_DATA)
    train_data = DataLoader(train_dataset, batch_size=32)

    valid_swb = SwitchboardCorpus(
        split_info=split_info["valid_split"], **cfg_dict["data"]
    )
    valid_dataset = BCDataset(valid_swb, BC_DATA)
    valid_data = DataLoader(valid_dataset, batch_size=32)

    # prepare model
    print("Build model ...")
    model = BCDAModel(
        cfg_dict["training"],
        cfg_dict["encoder"],
        cfg_dict["predictor"]
    )
    if init_state is not None:
        model.load_pretrained_vap(init_state)
    print("COMPLETE")

    # find best learning rate
    if not model.confg["optimizer"]["learning_rate"]:
        trainer = pl.Trainer(**cfg_dict["trainer"])
        lr_finder = trainer.tuner.lr_find(
            model,
            train_dataloaders=train_data,
            val_dataloaders=valid_data
        )
        model.confg["optimizer"]["learning_rate"] = lr_finder.suggestion()
        print("#" * 40)
        print("Initial LR: ", model.confg["optimizer"]["learning_rate"])
        print("#" * 40)

    # actual training
    trainer = pl.Trainer(
        callbacks=callbacks,
        **cfg_dict["trainer"],
    )
    trainer.fit(
        model,
        ckpt_path=ckpt_load,
        train_dataloaders=train_data,
        val_dataloaders=valid_data
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Interface to test a Voice Activity Projection model.",
        epilog="The following arguments are required: CONFIG CKPT_PATH"
    )

    parser.add_argument(
        "config", metavar="CONFIG", type=argparse.FileType("r"),
        help="YAML-file holding all data and model parameters, see data/conf for examples"
    )
    parser.add_argument(
        "--pretrained", metavar="CKPT_PATH", default=None,
        help="path to pretrained VAP model state"
    )
    parser.add_argument(
        "--load", metavar="CKPT_PATH", default=None,
        help="path to trained BCDA model state"
    )
    parser.add_argument(
        "--save", metavar="CKPT_NAME", default="checkpoint",
        help="name to save model state under"
    )


    if len(sys.argv) == 1:
        parser.print_help()
    else:
        args = parser.parse_args()
        cfg_dict = yaml.load(args.config, Loader=yaml.FullLoader)

        if args.pretrained is not None and args.load is not None:
            raise ValueError(
                "Pass either a pretrained VAP model OR a BCDA training state."
            )

        train(cfg_dict, args.pretrained, args.load, args.save)
