"""Helpers to run the learner on a TPU through torch_xla (``--device xla``).

Only the learner (gradient updates) goes to the TPU. Rollout workers and
validation processes stay on the CPU and receive CPU tensors, and every
checkpoint holds CPU tensors, so a run can move between cpu / cuda / xla.
torch_xla is imported lazily: the CPU path never needs it.
"""
import torch


def is_xla(device):
    return torch.device(device).type == "xla"


def resolve_device(name):
    """``torch.device`` for a --device value; "xla" needs torch_xla installed."""
    if name != "xla":
        return torch.device(name)
    try:
        import torch_xla
    except ImportError as error:
        raise RuntimeError("--device xla needs torch_xla (pip install torch_xla)") from error
    if hasattr(torch_xla, "device"):
        return torch_xla.device()
    import torch_xla.core.xla_model as xm
    return xm.xla_device()


def sync(device):
    """Run the pending lazy XLA graph; a no-op on other devices."""
    if not is_xla(device):
        return
    import torch_xla
    if hasattr(torch_xla, "sync"):
        torch_xla.sync()
    else:
        import torch_xla.core.xla_model as xm
        xm.mark_step()


def to_cpu(value):
    """Copy of ``value`` (tensors, nested dict/list/tuple) with tensors on the CPU,
    for checkpoints and for weights handed to CPU workers."""
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu", copy=True)
    if isinstance(value, dict):
        return {key: to_cpu(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(to_cpu(item) for item in value)
    return value
