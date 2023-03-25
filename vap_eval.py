"""test.py - Interface to test voice activity projection model"""
import argparse
import sys
import yaml

import torch

from vap.data_loader import SwitchboardCorpus
from vap.vap_model import VAPModel
from bcda.utils import BCDataset


def test(cfg_dict, model_state, test_file):
    # prepare data
    print("Load data ...")
    split_info = cfg_dict["data"].pop("split")

    swb = SwitchboardCorpus(
        split_info=split_info["test_split"], **cfg_dict["data"]
    )
    X = BCDataset(swb, test_file)
    print("COMPLETE")

    # prepare model
    print("Build model ...")
    model = VAPModel.load_from_checkpoint(model_state)
    model.freeze()
    print("COMPLETE")

    # evaluate model
    print("Evaluate model ...")
    conf_mtrx = dict()

    for i, x in enumerate(iter(X)):
        if i%100 == 0:
            print(i)
        outpt = torch.nn.functional.softmax(model(x)[0, -1], dim=0)

        # the BC should be uttered by the not active speaker
        is_bc = model.predictor.classification_head.is_backchannel(
            outpt, {"A": 1, "B": 0}[x["speakers"]]
        )
        conf_mtrx.setdefault((x["labels"] != "NO-BC", is_bc), 0)
        conf_mtrx[(x["labels"] != "NO-BC", is_bc)] += 1

    # print confusion matrix
    print(f"{'':25}{'Prediction':^40}")
    print(f"{'':25}{'BC':^20}{'NO-BC':^20}")
    print(f"{'Gold':<5}{'BC':>20}", end="")
    print(f"{conf_mtrx.get((True, True)):^20}{conf_mtrx.get((True, False)):^20}")
    print(f"{'':<5}{'NO-BC':>20}", end="")
    print(f"{conf_mtrx.get((False, True)):^20}{conf_mtrx.get((False, False)):^20}")


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
        "model", metavar="CKPT_PATH",
        help="path to model state to load for testing"
    )
    parser.add_argument(
        "--data", metavar="DATA_PATH",
        default="data/swb/utterance_is_backchannel.csv",
        help="path to plain BC record for testing"
    )

    if len(sys.argv) == 1:
        parser.print_help()
    else:
        args = parser.parse_args()
        cfg_dict = yaml.load(args.config, Loader=yaml.FullLoader)

        test(cfg_dict, args.model, args.data)
