#!/usr/bin/env python
import torch
import triton
import triton.language as tl

__all__ = [
    "mxfp4_fake_quant_block",
    "mxfp4_pack_uint4_to_uint8",
    "mxfp4_unpack_uint8_to_uint4",
    "mxfp4_dequantize",
    "mxfp4_quantize",
]

_TORCH_TO_TL_DTYPE = {
    torch.float32: tl.float32,
    torch.float: tl.float32,
    torch.float16: tl.float16,
    torch.half: tl.float16,
    torch.bfloat16: tl.bfloat16,
}

def _torch_dtype_to_tl(dtype: torch.dtype):
    if dtype not in _TORCH_TO_TL_DTYPE:
        raise ValueError(f"Unsupported dtype for mxfp4 fake quantization: {dtype}")
    return _TORCH_TO_TL_DTYPE[dtype]


@triton.jit
def mxfp4_fake_quant_kernel(
    x_ptr,
    y_ptr,
    M,
    N,
    stride_xm,
    stride_xn,
    stride_ym,
    stride_yn,
    BLOCK_SIZE: tl.constexpr,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    NUM_FP4_BLOCKS: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    row_start = pid_m * TILE_M
    col_start = pid_n * TILE_N

    x_block_ptr = tl.make_block_ptr(
        base=x_ptr, shape=(M, N), strides=(stride_xm, stride_xn),
        offsets=(row_start, col_start), block_shape=(TILE_M, TILE_N), order=(1, 0)
    )
    y_block_ptr = tl.make_block_ptr(
        base=y_ptr, shape=(M, N), strides=(stride_ym, stride_yn),
        offsets=(row_start, col_start), block_shape=(TILE_M, TILE_N), order=(1, 0)
    )

    tile = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
    tile_reshaped = tl.reshape(tile, (TILE_M, NUM_FP4_BLOCKS, BLOCK_SIZE))
    x_abs = tl.abs(tile_reshaped)

    block_amax = tl.max(x_abs, axis=2, keep_dims=True)

    descale = block_amax / 6.0
    descale = tl.maximum(descale, 1e-12)
    exp_fp = tl.ceil(tl.maximum(tl.log2(descale), -127.0))
    norm = tl.exp2(exp_fp)

    abs_norm = x_abs / norm

    """
    b0 = 0.25; b1 = 0.75; b2 = 1.25; b3 = 1.75; b4 = 2.5; b5 = 3.5; b6 = 5.0
    v0 = 0.0; v1 = 0.5; v2 = 1.0; v3 = 1.5; v4 = 2.0; v5 = 3.0; v6 = 4.0; v7 = 6.0

    ord = (abs_norm > b0) + (abs_norm > b1) + (abs_norm > b2) + (abs_norm > b3) + \
          (abs_norm > b4) + (abs_norm > b5) + (abs_norm > b6)
    """
    b0 = 0.25; b1 = 0.75; b2 = 1.25; b3 = 1.75; b4 = 2.5; b5 = 3.5; b6 = 5.0
    ord = (abs_norm > b0).to(tl.int32) + (abs_norm > b1).to(tl.int32) + (abs_norm > b2).to(tl.int32) + \
          (abs_norm > b3).to(tl.int32) + (abs_norm > b4).to(tl.int32) + (abs_norm > b5).to(tl.int32) + \
          (abs_norm > b6).to(tl.int32)
    v0 = 0.0; v1 = 0.5; v2 = 1.0; v3 = 1.5; v4 = 2.0; v5 = 3.0; v6 = 4.0; v7 = 6.0
    val = tl.where(ord == 0, v0,
          tl.where(ord == 1, v1,
          tl.where(ord == 2, v2,
          tl.where(ord == 3, v3,
          tl.where(ord == 4, v4,
          tl.where(ord == 5, v5,
          tl.where(ord == 6, v6, v7)))))))
    sign = tl.where(tile_reshaped >= 0, 1.0, -1.0)


    """
    val = tl.where(ord == 0, v0,
          tl.where(ord == 1, v1,
          tl.where(ord == 2, v2,
          tl.where(ord == 3, v3,
          tl.where(ord == 4, v4,
          tl.where(ord == 5, v5,
          tl.where(ord == 6, v6, v7)))))))

    sign = tl.where(tile_reshaped >= 0, 1.0, -1.0)
    """
    q_float = sign * val
    q_rescaled = q_float * norm

    tile_quant = tl.reshape(q_rescaled, (TILE_M, TILE_N))
    tl.store(y_block_ptr, tile_quant.to(OUT_DTYPE), boundary_check=(0, 1))


def mxfp4_fake_quant_block(
    x: torch.Tensor,
    block_size: int = 32,
    tile_rows: int = 16,
    tile_cols: int = 64,
    num_warps: int | None = None,
    num_stages: int | None = None,
) -> torch.Tensor:
    x_shape = x.shape
    x_dtype = x.dtype
    x = x.reshape(-1, x_shape[-1]).contiguous()

    M, N = x.shape
    y = torch.empty_like(x)
    # y = torch.empty((M, N), dtype=torch.float32, device=x.device).contiguous()

    stride_xm, stride_xn = x.stride()
    stride_ym, stride_yn = y.stride()

    tile_cols = max(tile_cols, block_size)
    tile_cols_aligned = ((tile_cols + block_size - 1) // block_size) * block_size
    num_fp4_blocks = tile_cols_aligned // block_size

    # grid = lambda *_: (triton.cdiv(M, tile_rows), triton.cdiv(N, tile_cols_aligned))
    grid = (triton.cdiv(M, tile_rows), triton.cdiv(N, tile_cols_aligned))

    launch_kwargs = {
        "BLOCK_SIZE": block_size,
        "TILE_M": tile_rows,
        "TILE_N": tile_cols_aligned,
        "NUM_FP4_BLOCKS": num_fp4_blocks,
        "OUT_DTYPE": _torch_dtype_to_tl(x_dtype),
    }
    if num_warps is not None:
        launch_kwargs["num_warps"] = num_warps
    if num_stages is not None:
        launch_kwargs["num_stages"] = num_stages

    mxfp4_fake_quant_kernel[grid](
        x, y, M, N, stride_xm, stride_xn, stride_ym, stride_yn, **launch_kwargs
    )

    y = y.view(*x_shape)
    return y


def mxfp4_pack_uint4_to_uint8(nibbles: torch.Tensor) -> torch.Tensor:
    left_side = nibbles[..., 0::2]
    right_side = nibbles[..., 1::2]
    packed = (right_side.clone() << 4)
    packed[..., : left_side.shape[-1]] |= left_side
    return packed


def mxfp4_unpack_uint8_to_uint4(packed: torch.Tensor) -> torch.Tensor:
    left_side = packed & 0x0F
    right_side = (packed >> 4) & 0x0F
    shape = list(packed.shape)
    shape[-1] = shape[-1] * 2
    result = torch.zeros(shape, dtype=torch.uint8, device=packed.device)
    result[..., 0::2] = left_side
    result[..., 1::2] = right_side
    return result


@triton.jit
def mxfp4_dequantize_kernel(
    packed_ptr,
    e8m0_exp_ptr,
    output_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    packed_start = pid * TILE_SIZE
    packed_offs = packed_start + tl.arange(0, TILE_SIZE)

    packed_row_idx = packed_offs // (N // 2)
    packed_col_idx = packed_offs % (N // 2)

    packed_mask = packed_col_idx < (N // 2)

    byte = tl.load(packed_ptr + packed_offs, mask=packed_mask, other=0)

    nib_low = byte & 0x0F
    nib_high = (byte >> 4) & 0x0F

    sign_low = 1.0 - 2.0 * ((nib_low & 0b1000) >> 3).to(tl.float32)
    sign_high = 1.0 - 2.0 * ((nib_high & 0b1000) >> 3).to(tl.float32)

    mag_low = (nib_low & 0b0111)
    mag_high = (nib_high & 0b0111)

    v0 = 0.0; v1 = 0.5; v2 = 1.0; v3 = 1.5; v4 = 2.0; v5 = 3.0; v6 = 4.0; v7 = 6.0

    val_low = tl.where(mag_low == 0, v0,
               tl.where(mag_low == 1, v1,
               tl.where(mag_low == 2, v2,
               tl.where(mag_low == 3, v3,
               tl.where(mag_low == 4, v4,
               tl.where(mag_low == 5, v5,
               tl.where(mag_low == 6, v6, v7)))))))

    val_high = tl.where(mag_high == 0, v0,
                tl.where(mag_high == 1, v1,
                tl.where(mag_high == 2, v2,
                tl.where(mag_high == 3, v3,
                tl.where(mag_high == 4, v4,
                tl.where(mag_high == 5, v5,
                tl.where(mag_high == 6, v6, v7)))))))

    val_low = sign_low * val_low
    val_high = sign_high * val_high

    out_col_low = packed_col_idx * 2
    out_col_high = out_col_low + 1
    out_offs_low = packed_row_idx * N + out_col_low
    out_offs_high = packed_row_idx * N + out_col_high

    block_col_low = out_col_low // BLOCK_SIZE
    block_col_high = out_col_high // BLOCK_SIZE
    exp_offs_low = packed_row_idx * (N // BLOCK_SIZE) + block_col_low
    exp_offs_high = packed_row_idx * (N // BLOCK_SIZE) + block_col_high

    exp_low_u8 = tl.load(e8m0_exp_ptr + exp_offs_low, mask=packed_mask & (out_col_low < N), other=127)
    exp_high_u8 = tl.load(e8m0_exp_ptr + exp_offs_high, mask=packed_mask & (out_col_high < N), other=127)

    scale_low = tl.exp2(exp_low_u8.to(tl.float32) - 127.0)
    scale_high = tl.exp2(exp_high_u8.to(tl.float32) - 127.0)

    result_low = val_low * scale_low
    result_high = val_high * scale_high

    out_mask_low = packed_mask & (out_col_low < N)
    out_mask_high = packed_mask & (out_col_high < N)
    tl.store(output_ptr + out_offs_low, result_low, mask=out_mask_low)
    tl.store(output_ptr + out_offs_high, result_high, mask=out_mask_high)


def mxfp4_dequantize(
    packed_tensor: torch.Tensor,
    e8m0_exp: torch.Tensor,
    block_size: int = 32,
    tile_size: int = 128,
    dtype: torch.dtype = torch.get_default_dtype(),
) -> torch.Tensor:
    packed_N = packed_tensor.shape[-1]
    N = packed_N * 2
    output_shape = list(packed_tensor.shape)
    output_shape[-1] = N
    output = torch.empty(output_shape, dtype=dtype, device=packed_tensor.device)

    grid = lambda meta: (triton.cdiv(packed_tensor.numel(), meta["TILE_SIZE"]),)

    mxfp4_dequantize_kernel[grid](
        packed_tensor, e8m0_exp, output, N,
        BLOCK_SIZE=block_size, TILE_SIZE=tile_size
    )
    return output

@triton.jit
def mxfp4_quantize_kernel(
    x_ptr,
    q_ptr,
    exp_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    stride_xm: tl.constexpr,
    stride_xn: tl.constexpr,
    stride_qm: tl.constexpr,
    stride_qn: tl.constexpr,
    stride_em: tl.constexpr,   # exp strides: (M, N//BLOCK_SIZE)
    stride_en: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    NUM_FP4_BLOCKS: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    row_start = pid_m * TILE_M
    col_start = pid_n * TILE_N

    # Load tile [TILE_M, TILE_N]
    x_block_ptr = tl.make_block_ptr(
        base=x_ptr,
        shape=(M, N),
        strides=(stride_xm, stride_xn),
        offsets=(row_start, col_start),
        block_shape=(TILE_M, TILE_N),
        order=(1, 0),
    )
    x_tile = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)

    # Reshape to [TILE_M, NUM_FP4_BLOCKS, BLOCK_SIZE]
    x_r = tl.reshape(x_tile, (TILE_M, NUM_FP4_BLOCKS, BLOCK_SIZE))
    x_abs = tl.abs(x_r)

    # amax per block
    block_amax = tl.max(x_abs, axis=2, keep_dims=True)  # [TILE_M, NUM_FP4_BLOCKS, 1]
    descale = block_amax / 6.0

    # ---- strict torch-like handling for descale == 0 ----
    # torch: log2(0)=-inf -> max(-inf, -127)=-127 -> ceil(-127)=-127
    neg_inf = -float("inf")
    log2_descale = tl.where(descale == 0.0, neg_inf, tl.log2(descale))
    exp_fp = tl.ceil(tl.maximum(log2_descale, -127.0))  # float exponent
    # -----------------------------------------------

    norm = tl.exp2(exp_fp)

    # normalize for fp4 rounding
    abs_norm = x_abs / norm

    # ord by bounds (E2M1)
    b0 = 0.25; b1 = 0.75; b2 = 1.25; b3 = 1.75; b4 = 2.5; b5 = 3.5; b6 = 5.0
    ord = (abs_norm > b0).to(tl.int32) + (abs_norm > b1).to(tl.int32) + (abs_norm > b2).to(tl.int32) + \
          (abs_norm > b3).to(tl.int32) + (abs_norm > b4).to(tl.int32) + (abs_norm > b5).to(tl.int32) + \
          (abs_norm > b6).to(tl.int32)  # 0..7

    # ---- strict torch.sign-compatible sign_bit ----
    # torch.sign: x>0 => 1, x<0 => -1, x==0 => 0
    # sign_bit = (2 - sign)//2, so:
    #   x>0 ->0
    #   x<0 ->1
    #   x==0 ->1   (this is why all-zero maps to code 8)
    sign_f = tl.where(x_r > 0.0, 1.0, tl.where(x_r < 0.0, -1.0, 0.0))
    sign_i = sign_f.to(tl.int32)
    sign_bit = (2 - sign_i) // 2
    # ---------------------------------------------

    code = (sign_bit * 8 + ord).to(tl.uint8)  # 0..15
    code_tile = tl.reshape(code, (TILE_M, TILE_N))

    # store q
    q_block_ptr = tl.make_block_ptr(
        base=q_ptr,
        shape=(M, N),
        strides=(stride_qm, stride_qn),
        offsets=(row_start, col_start),
        block_shape=(TILE_M, TILE_N),
        order=(1, 0),
    )
    tl.store(q_block_ptr, code_tile, boundary_check=(0, 1))

    # store exp per-block: (exp_fp + 127) as uint8
    exp_i = exp_fp.to(tl.int32)
    exp_i = tl.maximum(exp_i, -127)
    exp_i = tl.minimum(exp_i, 127)
    exp_u8 = (exp_i + 127).to(tl.uint8)  # [TILE_M, NUM_FP4_BLOCKS, 1]

    exp_row = row_start + tl.arange(0, TILE_M)                         # [TILE_M]
    exp_col = (col_start // BLOCK_SIZE) + tl.arange(0, NUM_FP4_BLOCKS) # [NUM_FP4_BLOCKS]
    m_mask = exp_row < M
    n_mask = exp_col < (N // BLOCK_SIZE)
    mask2d = m_mask[:, None] & n_mask[None, :]

    exp_u8_2d = tl.reshape(exp_u8, (TILE_M, NUM_FP4_BLOCKS))
    exp_ptrs = exp_ptr + exp_row[:, None] * stride_em + exp_col[None, :] * stride_en
    tl.store(exp_ptrs, exp_u8_2d, mask=mask2d)


def mxfp4_quantize(
    x: torch.Tensor,
    block_size: int = 32,
    tile_rows: int = 16,
    tile_cols: int = 128,
    num_warps: int | None = None,
    num_stages: int | None = None,
):
    """
    No-pack quantize:
      returns:
        q: uint8 tensor same shape as x (each element stores fp4 code 0..15; no packing)
        e8m0_scale: uint8 tensor shape [M, N//block_size] then viewed as [q.shape[0], -1]
    """
    x_shape = x.shape
    x2 = x.reshape(-1, x_shape[-1]).contiguous()
    M, N = x2.shape
    assert N % block_size == 0, f"Last dim N={N} must be divisible by block_size={block_size}"

    q = torch.empty((M, N), dtype=torch.uint8, device=x.device)
    e8m0 = torch.empty((M, N // block_size), dtype=torch.uint8, device=x.device)

    stride_xm, stride_xn = x2.stride()
    stride_qm, stride_qn = q.stride()
    stride_em, stride_en = e8m0.stride()

    tile_cols = max(tile_cols, block_size)
    tile_cols_aligned = ((tile_cols + block_size - 1) // block_size) * block_size
    num_fp4_blocks = tile_cols_aligned // block_size

    grid = (triton.cdiv(M, tile_rows), triton.cdiv(N, tile_cols_aligned))

    launch_kwargs = dict(
        M=M, N=N,
        stride_xm=stride_xm, stride_xn=stride_xn,
        stride_qm=stride_qm, stride_qn=stride_qn,
        stride_em=stride_em, stride_en=stride_en,
        BLOCK_SIZE=block_size,
        TILE_M=tile_rows,
        TILE_N=tile_cols_aligned,
        NUM_FP4_BLOCKS=num_fp4_blocks,
    )
    if num_warps is not None:
        launch_kwargs["num_warps"] = num_warps
    if num_stages is not None:
        launch_kwargs["num_stages"] = num_stages

    mxfp4_quantize_kernel[grid](x2, q, e8m0, **launch_kwargs)

    q = q.view(*x_shape)
    e8m0 = e8m0.view(q.shape[0], -1)
    return q, e8m0
