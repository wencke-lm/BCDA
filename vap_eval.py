"""vap_eval.py - Interface to test voice activity projection model"""
import argparse
import sys
import yaml

from sklearn.metrics import confusion_matrix, f1_score
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

    true = []
    pred = []

    for i, x in enumerate(iter(X)):
        if i%100 == 0:
            print(i)

        outpt = torch.nn.functional.softmax(model(x)[0, -1], dim=0)

        # the BC should be uttered by the not active speaker
        is_bc = model.predictor.classification_head.is_backchannel(
            outpt, {"A": 1, "B": 0}[x["speakers"]]
        )
        true.append(x["labels"] != "NO-BC")
        pred.append(is_bc)

    conf_mtrx = confusion_matrix(true, pred, labels=[False, True])

    # print confusion matrix
    print(f"{'':25}{'Prediction':^40}")
    print(f"{'':25}{'NO-BC':^20}{'BC':^20}")
    print(f"{'Gold':<5}{'NO-BC':>20}", end="")
    print(f"{conf_mtrx[0, 0]:^20}{conf_mtrx[0, 1]:^20}")
    print(f"{'':<5}{'BC':>20}", end="")
    print(f"{conf_mtrx[1, 0]:^20}{conf_mtrx[1, 1]:^20}")

    f1_no_bc, f1_bc = f1_score(
        true, pred, average=None, labels=[False, True]
    )
    weighted_f1 = f1_score(
        true, pred, average="weighted", labels=[False, True]
    )

    print("NO-BC F1:", f1_no_bc)
    print("BC F1:", f1_bc)
    print("=> Weighted F1:", weighted_f1)


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