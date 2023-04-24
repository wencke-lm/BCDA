"""bcda_eval.py - Interface to test BC prediction model"""
import argparse
import os
import sys
import yaml

from sklearn.metrics import confusion_matrix, f1_score
import torch
from torch.utils.data import DataLoader

from bpm_model import BPM_MT
from bpm_utils import BCDataset


def test(model_state, test_file):
    path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    # prepare data
    print("Load data ...")
    X = BCDataset(
        os.path.join(path, "data", "swb", "conversations.eval"),
        os.path.join(
            path, "data", "swb",
            "utterance_is_backchannel_with_sentiment_with_context.csv"
        ),
        os.path.join(path, "data", "swb", "swb_audios")
    )
    print("COMPLETE")

    # prepare model
    print("Build model ...")
    model = BPM_MT("bert-base-uncased").load_from_checkpoint(model_state)
    model.freeze()
    print("COMPLETE")

    # evaluate model
    print("Evaluate model ...")

    true = []
    pred = []

    for i, x in enumerate(iter(X)):
        if i%100 == 0:
            print(i)

        main_output, sub_output = model(
            x["input_ids"], x["attention_mask"],
            x["acoustic_input"], train=True
        )
        outpt = int(torch.argmax(main_output, dim=1))

        true.append(model.label_to_idx[x["main_labels"]])
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
        "model", metavar="CKPT_PATH",
        help="path to model state to load for testing"
    )
    parser.add_argument(
        "--data", metavar="DATA_PATH",
        default="data/swb/utterance_is_backchannel_with_sentiment_with_context.csv",
        help="path to plain BC record for testing"
    )

    if len(sys.argv) == 1:
        parser.print_help()
    else:
        args = parser.parse_args()

        test(args.model, args.data)
