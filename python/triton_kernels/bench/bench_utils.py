from copy import deepcopy
from dataclasses import dataclass

import triton_kernels.swiglu
from triton_kernels.matmul import FlexCtx, FusedActivation, PrecisionConfig, FnSpecs
from triton_kernels.numerics import InFlexData
from triton_kernels.numerics_details.mxfp import downcast_to_mxfp
from triton_kernels.tensor import FP4, Tensor, convert_layout, wrap_torch_tensor
from triton_kernels.target_info import cuda_capability_geq, get_cdna_version, is_cuda, is_hip
from triton_kernels.tensor_details import layout
import torch


def _quantize_weight(w, dtype, **opt):
    if dtype == "bf16":
        # keep weight layout stable after casting
        wq = w.to(torch.bfloat16).transpose(-1, -2).contiguous().transpose(-1, -2)
        return wq, InFlexData(), None
    if dtype == "fp8":
        fp8e4_dtype = torch.float8_e4m3fn if get_cdna_version() != 3 else torch.float8_e4m3fnuz
        wq = w.to(fp8e4_dtype)
        if is_cuda() and not cuda_capability_geq(10, 0):
            # pre-transpose for older cuda to match kernel expectations
            wq = wq.transpose(-1, -2).contiguous().transpose(-1, -2)
        return wq, InFlexData(dtype=wq.dtype, scale=w.abs().max().unsqueeze(0)), None

    assert dtype == "mx4", f"{dtype=}"
    wq, w_scale = downcast_to_mxfp(w.to(torch.bfloat16), torch.uint8, axis=-1)
    if opt:
        if "value_layout" in opt:
            w = convert_layout(w, opt["value_layout"], **opt["value_layout_opts"])
        if "scale_layout" in opt:
            w_scale = convert_layout(wrap_torch_tensor(w_scale), opt["scale_layout"], **opt["scale_layout_opts"])
    return w, InFlexData(), w_scale


def _quantize_activation(x, dtype=None, **opt):
    if x is None or dtype is None or "mx" not in dtype:
        return x, InFlexData(), None
    assert dtype == "mx8", f"{dtype=}"
    fp8_dtype = torch.float8_e4m3fn if get_cdna_version() != 3 else torch.float8_e4m3fnuz
    xq, x_scale = downcast_to_mxfp(x.to(torch.bfloat16), fp8_dtype, axis=-1)
    if opt:
        if "value_layout" in opt:
            xq = convert_layout(wrap_torch_tensor(xq, dtype=xq.dtype), opt["value_layout"], **opt["value_layout_opts"])
        if "scale_layout" in opt:
            x_scale = convert_layout(wrap_torch_tensor(x_scale), opt["scale_layout"], **opt["scale_layout_opts"])
    return xq, InFlexData(), x_scale


@dataclass
class MlpNumerics:
    x: torch.Tensor | Tensor | None
    wg: torch.Tensor | Tensor | None
    w1: torch.Tensor | Tensor | None
    w2: torch.Tensor | Tensor | None
    pcg: PrecisionConfig
    pc1: PrecisionConfig
    pc2: PrecisionConfig
    activation: FusedActivation


def _make_default_mlp_activation() -> FusedActivation:
    return FusedActivation(
        FnSpecs("swiglu", triton_kernels.swiglu.swiglu_fn, ("alpha", "limit"), reduction_n=2),
        (1.0, 1.0),
    )


def _make_mx4_quantization_opts(batch: int, dtype: str) -> dict:
    if dtype != "mx4" or is_hip():
        return {}
    num_warps = 4 if batch <= 512 and cuda_capability_geq(10, 0) else 8
    value_layout, value_layout_opts = layout.make_default_matmul_mxfp4_w_layout(mx_axis=1)
    scale_layout, scale_layout_opts = layout.make_default_matmul_mxfp4_w_scale_layout(mx_axis=1, num_warps=num_warps)
    return {
        "value_layout": value_layout,
        "value_layout_opts": value_layout_opts,
        "scale_layout": scale_layout,
        "scale_layout_opts": scale_layout_opts,
    }


def _make_mx8_quantization_opts(dtype: str) -> dict:
    if dtype != "mx8" or is_hip():
        return {}
    scale_layout, scale_layout_opts = layout.make_default_matmul_mxfp8_act_scale_layout()
    return {
        "scale_layout": scale_layout,
        "scale_layout_opts": scale_layout_opts,
    }


def prepare_mlp_numerics(batch: int, w_dtype: str, wg, w1, w2, *, x=None, x_dtype: str | None = None) -> MlpNumerics:
    """
    Quantize weights (and optionally activations) for the MLP benchmark.
    When x/x_dtype specify mx8, x is downcast to mxfp8 and a_mx_scale is set for the gate matmul (a=mx8).
    Weights using w_dtype=\"mx4\" are downcast to mxfp4 with the corresponding b_mx_scale.
    """
    w_quant_opts = _make_mx4_quantization_opts(batch, w_dtype)
    act_quant_opts = _make_mx8_quantization_opts(x_dtype or "")

    x, x_flex, x_scale = _quantize_activation(x, x_dtype, **deepcopy(act_quant_opts))
    wg, wg_flex, wg_scale = _quantize_weight(wg, "bf16")
    w1, w1_flex, w1_scale = _quantize_weight(w1, w_dtype, **deepcopy(w_quant_opts))
    w2, w2_flex, w2_scale = _quantize_weight(w2, w_dtype, **deepcopy(w_quant_opts))

    activation = _make_default_mlp_activation()
    return MlpNumerics(
        x=x,
        wg=wg,
        w1=w1,
        w2=w2,
        pcg=PrecisionConfig(flex_ctx=FlexCtx(rhs_data=wg_flex), a_mx_scale=x_scale, b_mx_scale=wg_scale),
        pc1=PrecisionConfig(flex_ctx=FlexCtx(lhs_data=x_flex, rhs_data=w1_flex), b_mx_scale=w1_scale),
        pc2=PrecisionConfig(flex_ctx=FlexCtx(rhs_data=w2_flex), b_mx_scale=w2_scale),
        activation=activation,
    )


def resolve_x_dtype(x_dtype: str) -> torch.dtype:
    dtype_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp8": torch.float8_e4m3fn,
        "mx8": torch.float8_e4m3fn,
    }
    dtype = dtype_map[x_dtype]
    if dtype == torch.float8_e4m3fn and get_cdna_version() == 3:
        return torch.float8_e4m3fnuz
    return dtype
