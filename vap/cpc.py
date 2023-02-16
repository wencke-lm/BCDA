"""cpc.py - Contrastive predictive wave encoding model"""
from cpc.cpc_default_config import get_default_cpc_config
from cpc.feature_loader import getEncoder, getAR
from cpc.model import CPCModel as cpcmodel

import torch


def load_CPC(checkpoint_origin=None, device="cpu"):
    """Model trained using contrastive predictive coding.

    Details at: https://arxiv.org/pdf/2002.02848.pdf

    Arguments:
        checkpoint_origin (str):
            Either url to a downloadable checkpoint or
            path to a locally saved checkpoint.
            If None, clean state is initialised.
            Defaults to None.
        device (str):
            Storage location to place the model into.
            Defaults to cpu.

    Returns:
        CPCModel:
            Takes inputs of shape (N_Batch, 1, K_Frames)

    """
    config = get_default_cpc_config()

    if checkpoint_origin is not None:
        if checkpoint_origin.startswith("http"):
            checkpoint = torch.hub.load_state_dict_from_url(
                checkpoint_origin,
                map_location=device
            )
        else:
            checkpoint = torch.load(
                checkpoint_origin,
                map_location=device
            )

        for k, v in checkpoint["config"].items():
            setattr(config, k, v)

    encoder_net = getEncoder(config)
    ar_net = getAR(config)

    model = cpcmodel(encoder_net, ar_net)
    model.name = "cpc"

    if checkpoint_origin is not None:
        model.load_state_dict(checkpoint["weights"], strict=False)

    return model
