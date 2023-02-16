"""train.py - Interface to train voice activity projection model"""
import argparse
import sys

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    StochasticWeightAveraging,
)
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader
import yaml

from vap.data_loader import SwitchboardCorpus
from vap.vap_model import VAPModel


def train(cfg_dict, ckpt_load, ckpt_save):
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

    # prepare data and model
    print("Load data ...")
    swb = SwitchboardCorpus(**cfg_dict["data"])
    data = DataLoader(swb, batch_size=16)
    print("COMPLETE")

    print("Build model ...")
    model = VAPModel(cfg_dict["training"], cfg_dict["encoder"], cfg_dict["predictor"])
    print("COMPLETE")

    # # find best learning rate
    # trainer = pl.Trainer(accelerator="gpu")
    # lr_finder = trainer.tuner.lr_find(model, train_dataloaders=data)
    # model.confg["optimizer"]["learning_rate"] = lr_finder.suggestion()
    # print("#" * 40)
    # print("Initial Learning Rate: ", model.confg["optimizer"]["learning_rate"])
    # print("#" * 40)

    # actual training
    trainer = pl.Trainer(
        callbacks=callbacks,
        **cfg_dict["trainer"],
    )
    trainer.fit(model, ckpt_path=ckpt_load, train_dataloaders=data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Interface to train a Voice Activity Projection model.",
        epilog="The following arguments are required: CONFIG"
    )

    parser.add_argument(
        "config", metavar="CONFIG", type=argparse.FileType("r"),
        help="YAML-file holding all training and model parameters, see data/conf for examples"
    )
    parser.add_argument(
        "--load", metavar="CKPT_PATH", default=None,
        help="path to model state to start training from or special keyword 'last'"
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

        train(cfg_dict, args.load, args.save)
