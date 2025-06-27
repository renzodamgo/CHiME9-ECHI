import auraloss
import torch

EPS = 1e-5


def get_loss(name, params=None):
    """
    Get the loss function for the specified architecture.

    Parameters
    ----------
    name : str
        The name of the loss function.
    params : dict
        The parameters for the loss function.

    Returns
    -------
    torch.nn.Module
        The loss function for the specified architecture.
    """
    if name == "sisnr":
        return auraloss.time.SISDRLoss(eps=EPS)
    elif name == "spec":
        return auraloss.freq.STFTLoss(**params)
    elif name == "multispec":
        return auraloss.freq.MultiResolutionSTFTLoss(**params)
    else:
        raise ValueError(f"Unknown loss name: {name}")


def get_lrmethod(name: str, optim, params):
    if name == "plateau_reduce":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optim, **params)
    else:
        raise ValueError(f"LR Scheduler {name} not implemented. Add code here")
