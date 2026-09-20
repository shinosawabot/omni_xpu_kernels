"""Standalone scaled dot-product attention kernel wrapper."""

import torch

from .. import __xpu_target__
from .. import _compile_meta as _meta
from .._compile_ops import compile_op


def _get_native():
    from .. import _load_extension
    return _load_extension().sdp


def _validate_torch_sdpa_inputs(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    for tensor, name in ((q, "q"), (k, "k"), (v, "v")):
        if tensor.device.type != "xpu":
            raise RuntimeError(f"{name} must be on XPU")
        if not tensor.is_contiguous():
            raise RuntimeError(f"{name} must be contiguous")
        if tensor.dim() != 4:
            raise RuntimeError(f"{name} must be 4-D [B, L, H, D]")
        if tensor.size(0) != 1:
            raise RuntimeError(f"{name} batch size must be 1")
        if tensor.size(3) not in (64, 128):
            raise RuntimeError(f"{name} head_dim must be 64 or 128")
        if tensor.dtype not in (torch.float16, torch.bfloat16):
            raise RuntimeError(f"{name} dtype must be FP16 or BF16")

    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise RuntimeError("q, k, v must have the same dtype")
    if k.size(1) != v.size(1):
        raise RuntimeError("k and v must have the same sequence length")
    if k.size(2) != v.size(2):
        raise RuntimeError("k and v must have the same head count")
    if q.size(2) != k.size(2):
        raise RuntimeError("q, k, v must have the same head count")
    if q.size(3) != k.size(3) or q.size(3) != v.size(3):
        raise RuntimeError("q, k, v must have the same head_dim")


def _torch_xpu_sdp(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Use PyTorch's native XPU SDPA when no DG2 sidecar is available."""
    _validate_torch_sdpa_inputs(q, k, v)
    out = torch.nn.functional.scaled_dot_product_attention(
        q.permute(0, 2, 1, 3).contiguous(),
        k.permute(0, 2, 1, 3).contiguous(),
        v.permute(0, 2, 1, 3).contiguous(),
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
    )
    return out.permute(0, 2, 1, 3).contiguous()


@compile_op("sdp_sdp", _meta.unchanged, ordered=True)
def sdp(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    if __xpu_target__ == "dg2":
        return _torch_xpu_sdp(q, k, v)
    if torch.compiler.is_compiling():
        return torch.ops.omni_xpu.sdp_sdp(q, k, v)
    return _get_native().sdp(q, k, v)


__all__ = ["sdp"]
