"""bcda_eval.py - Interface to test BC prediction model"""
import argparse
import sys
import yaml

from sklearn.metrics import confusion_matrix, f1_score
import torch

from vap.data_loader import SwitchboardCorpus
from bcda.bcda_model import BCDAModel
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
    model = BCDAModel.load_from_checkpoint(model_state, strict=False)
    model.to("cuda:0" if torch.cuda.is_available() else "cpu")
    model.freeze()
    print(model.training)
    print("COMPLETE")

    # evaluate model
    print("Evaluate model ...")

    true = []
    pred = []

    for i, x in enumerate(iter(X)):
        if i%100 == 0:
            print(i)

        for k, v in x.items():
            if torch.is_tensor(v):
                x[k] = v.to("cuda:0" if torch.cuda.is_available() else "cpu")

        outpt = int(torch.argmax(model(x)[0], dim=1))

        true.append(model.label_to_idx[x["labels"]])
        pred.append(outpt)

    conf_mtrx = confusion_matrix(true, pred, labels=[0, 1, 2])

    # print confusion matrix
    print(f"{'':25}{'Prediction':^40}")
    print(f"{'':25}{'NO-BC':^20}{'CONTINUER':^20}{'ASSESSMENT':^20}")
    print(f"{'Gold':<5}{'NO-BC':>20}", end="")
    print(f"{conf_mtrx[0, 0]:^20}{conf_mtrx[0, 1]:^20}{conf_mtrx[0, 2]:^20}")
    print(f"{'':<5}{'CONTINUER':>20}", end="")
    print(f"{conf_mtrx[1, 0]:^20}{conf_mtrx[1, 1]:^20}{conf_mtrx[1, 2]:^20}")
    print(f"{'':<5}{'ASSESSMENT':>20}", end="")
    print(f"{conf_mtrx[2, 0]:^20}{conf_mtrx[2, 1]:^20}{conf_mtrx[2, 2]:^20}")

    f1_no_bc, f1_cont, f1_ass =  f1_score(
        true, pred, average=None, labels=[0, 1, 2]
    )
    weighted_f1 = f1_score(
        true, pred, average="weighted", labels=[0, 1, 2]
    )

    print("NO-BC F1:", f1_no_bc)
    print("CONTINUER F1:", f1_cont)
    print("ASSESSMENT F1:", f1_ass)
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
        default="data/swb/utterance_is_backchannel_with_da_with_context.csv",
        help="path to plain BC record for testing"
    )

    if len(sys.argv) == 1:
        parser.print_help()
    else:
        args = parser.parse_args()
        cfg_dict = yaml.load(args.config, Loader=yaml.FullLoader)

        test(cfg_dict, args.model, args.data)
