import torch
from loguru import logger


def get_device():
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        logger.info("MPS backend detected, using MPS")
        return "mps"
    elif torch.cuda.is_available():
        logger.info("CUDA detected, using CUDA")
        return "cuda"
    else:
        logger.info("No GPU detected, using CPU")
        return "cpu"


DEVICE = get_device()
