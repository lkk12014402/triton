# Triton 实战示例集

> 常见操作的 Triton 实现参考代码

## 目录

1. [基础操作](#基础操作)
2. [融合算子](#融合算子)
3. [归约操作](#归约操作)
4. [矩阵运算](#矩阵运算)
5. [激活函数](#激活函数)
6. [规范化层](#规范化层)

---

## 基础操作

### 1. 元素级加法（带掩码）

```python
import torch
import triton
import triton.language as tl

@triton.jit
def elementwise_add_kernel(
    x_ptr, y_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    output = x + y
    
    tl.store(output_ptr + offsets, output, mask=mask)

def elementwise_add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    elementwise_add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output
```

### 2. 标量乘法

```python
@triton.jit
def scalar_mul_kernel(
    x_ptr, output_ptr,
    scalar,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    output = x * scalar
    
    tl.store(output_ptr + offsets, output, mask=mask)
```

### 3. 条件选择（Where）

```python
@triton.jit
def where_kernel(
    condition_ptr, x_ptr, y_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    condition = tl.load(condition_ptr + offsets, mask=mask)
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    output = tl.where(condition, x, y)
    
    tl.store(output_ptr + offsets, output, mask=mask)
```

---

## 融合算子

### 1. 融合乘加（FMA：x * y + z）

```python
@triton.jit
def fused_multiply_add_kernel(
    x_ptr, y_ptr, z_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    z = tl.load(z_ptr + offsets, mask=mask)
    
    output = x * y + z  # 融合为单次内核调用
    
    tl.store(output_ptr + offsets, output, mask=mask)

def fused_multiply_add(x, y, z):
    output = torch.empty_like(x)
    n = output.numel()
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    fused_multiply_add_kernel[grid](x, y, z, output, n, BLOCK_SIZE=1024)
    return output
```

### 2. 融合线性层 + ReLU

```python
@triton.jit
def fused_linear_relu_kernel(
    x_ptr, weight_ptr, bias_ptr, output_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = weight_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        x = tl.load(x_ptrs)
        w = tl.load(w_ptrs)
        accumulator += tl.dot(x, w)
        x_ptrs += BLOCK_SIZE_K * stride_xk
        w_ptrs += BLOCK_SIZE_K * stride_wk
    
    # 加偏置
    bias = tl.load(bias_ptr + offs_n)
    accumulator += bias[None, :]
    
    # 应用 ReLU
    output = tl.maximum(accumulator, 0)
    
    # 存储
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    output_ptrs = output_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(output_ptrs, output, mask=mask)
```

---

## 归约操作

### 1. 向量求和

```python
@triton.jit
def sum_kernel(
    x_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # 块内归约
    sum_val = tl.sum(x)
    
    # 使用原子操作累加到全局结果
    tl.atomic_add(output_ptr, sum_val)

def vector_sum(x):
    output = torch.zeros(1, device=x.device, dtype=x.dtype)
    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    sum_kernel[grid](x, output, n, BLOCK_SIZE=1024)
    return output
```

### 2. 最大值

```python
@triton.jit
def max_kernel(
    x_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=float('-inf'))
    
    # 块内最大值
    max_val = tl.max(x)
    
    # 原子最大值（需要循环实现）
    tl.atomic_max(output_ptr, max_val)
```

### 3. 按行求和（2D）

```python
@triton.jit
def rowsum_kernel(
    x_ptr, output_ptr,
    M, N,
    stride_xm, stride_xn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    
    mask_m = offs_m < M
    
    row_sum = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    
    for n in range(0, tl.cdiv(N, BLOCK_SIZE_N)):
        offs_n_curr = n * BLOCK_SIZE_N + offs_n
        mask = mask_m[:, None] & (offs_n_curr[None, :] < N)
        
        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + offs_n_curr[None, :] * stride_xn,
            mask=mask,
            other=0.0
        )
        
        row_sum += tl.sum(x, axis=1)
    
    tl.store(output_ptr + offs_m, row_sum, mask=mask_m)
```

---

## 矩阵运算

### 1. 矩阵转置

```python
@triton.jit
def transpose_kernel(
    x_ptr, output_ptr,
    M, N,
    stride_xm, stride_xn,
    stride_om, stride_on,
    BLOCK_SIZE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    
    x = tl.load(
        x_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn,
        mask=mask
    )
    
    # 转置：交换维度
    tl.store(
        output_ptr + offs_n[:, None] * stride_om + offs_m[None, :] * stride_on,
        tl.trans(x),
        mask=tl.trans(mask)
    )
```

### 2. 批量矩阵乘法（简化版）

```python
@triton.jit
def batched_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    batch_size, M, N, K,
    stride_ab, stride_am, stride_ak,
    stride_bb, stride_bk, stride_bn,
    stride_cb, stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)
    
    # 计算批次偏移
    a_batch_offset = pid_b * stride_ab
    b_batch_offset = pid_b * stride_bb
    c_batch_offset = pid_b * stride_cb
    
    # 矩阵乘法逻辑（类似标准 matmul）
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    a_ptrs = a_ptr + a_batch_offset + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + b_batch_offset + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    c = accumulator.to(tl.float16)
    
    c_ptrs = c_ptr + c_batch_offset + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)
```

---

## 激活函数

### 1. ReLU

```python
@triton.jit
def relu_kernel(
    x_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    output = tl.maximum(x, 0)
    
    tl.store(output_ptr + offsets, output, mask=mask)
```

### 2. GELU（高斯误差线性单元）

```python
@triton.jit
def gelu_kernel(
    x_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # GELU 近似：0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    sqrt_2_over_pi = 0.7978845608028654
    x_cubed = x * x * x
    inner = sqrt_2_over_pi * (x + 0.044715 * x_cubed)
    tanh_inner = tl.tanh(inner)
    output = 0.5 * x * (1.0 + tanh_inner)
    
    tl.store(output_ptr + offsets, output, mask=mask)
```

### 3. Sigmoid

```python
@triton.jit
def sigmoid_kernel(
    x_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # sigmoid(x) = 1 / (1 + exp(-x))
    output = 1.0 / (1.0 + tl.exp(-x))
    
    tl.store(output_ptr + offsets, output, mask=mask)
```

### 4. SiLU / Swish

```python
@triton.jit
def silu_kernel(
    x_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
    output = x / (1.0 + tl.exp(-x))
    
    tl.store(output_ptr + offsets, output, mask=mask)
```

---

## 规范化层

### 1. Softmax（按行）

```python
@triton.jit
def softmax_kernel(
    x_ptr, output_ptr,
    M, N,
    stride_xm, stride_xn,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    offs_m = pid
    offs_n = tl.arange(0, BLOCK_SIZE)
    
    # 加载一行数据
    x_row = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    max_val = float('-inf')
    
    # 第一遍：找最大值（数值稳定性）
    for n in range(0, tl.cdiv(N, BLOCK_SIZE)):
        offs_n_curr = n * BLOCK_SIZE + offs_n
        mask = offs_n_curr < N
        
        x = tl.load(
            x_ptr + offs_m * stride_xm + offs_n_curr * stride_xn,
            mask=mask,
            other=float('-inf')
        )
        max_val = tl.maximum(max_val, tl.max(x))
    
    # 第二遍：计算 exp 和求和
    sum_exp = 0.0
    for n in range(0, tl.cdiv(N, BLOCK_SIZE)):
        offs_n_curr = n * BLOCK_SIZE + offs_n
        mask = offs_n_curr < N
        
        x = tl.load(
            x_ptr + offs_m * stride_xm + offs_n_curr * stride_xn,
            mask=mask,
            other=0.0
        )
        exp_x = tl.exp(x - max_val)
        sum_exp += tl.sum(tl.where(mask, exp_x, 0.0))
    
    # 第三遍：归一化并存储
    for n in range(0, tl.cdiv(N, BLOCK_SIZE)):
        offs_n_curr = n * BLOCK_SIZE + offs_n
        mask = offs_n_curr < N
        
        x = tl.load(
            x_ptr + offs_m * stride_xm + offs_n_curr * stride_xn,
            mask=mask,
            other=0.0
        )
        output = tl.exp(x - max_val) / sum_exp
        
        tl.store(
            output_ptr + offs_m * stride_xm + offs_n_curr * stride_xn,
            output,
            mask=mask
        )
```

### 2. Layer Normalization

```python
@triton.jit
def layernorm_kernel(
    x_ptr, weight_ptr, bias_ptr, output_ptr,
    M, N,
    stride_xm, stride_xn,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    offs_m = pid
    offs_n = tl.arange(0, BLOCK_SIZE)
    
    # 第一遍：计算均值
    mean = 0.0
    for n in range(0, tl.cdiv(N, BLOCK_SIZE)):
        offs_n_curr = n * BLOCK_SIZE + offs_n
        mask = offs_n_curr < N
        
        x = tl.load(
            x_ptr + offs_m * stride_xm + offs_n_curr * stride_xn,
            mask=mask,
            other=0.0
        )
        mean += tl.sum(tl.where(mask, x, 0.0))
    mean = mean / N
    
    # 第二遍：计算方差
    var = 0.0
    for n in range(0, tl.cdiv(N, BLOCK_SIZE)):
        offs_n_curr = n * BLOCK_SIZE + offs_n
        mask = offs_n_curr < N
        
        x = tl.load(
            x_ptr + offs_m * stride_xm + offs_n_curr * stride_xn,
            mask=mask,
            other=0.0
        )
        diff = tl.where(mask, x - mean, 0.0)
        var += tl.sum(diff * diff)
    var = var / N
    
    # 第三遍：归一化
    rstd = 1.0 / tl.sqrt(var + eps)
    
    for n in range(0, tl.cdiv(N, BLOCK_SIZE)):
        offs_n_curr = n * BLOCK_SIZE + offs_n
        mask = offs_n_curr < N
        
        x = tl.load(
            x_ptr + offs_m * stride_xm + offs_n_curr * stride_xn,
            mask=mask,
            other=0.0
        )
        weight = tl.load(weight_ptr + offs_n_curr, mask=mask, other=1.0)
        bias = tl.load(bias_ptr + offs_n_curr, mask=mask, other=0.0)
        
        normalized = (x - mean) * rstd
        output = normalized * weight + bias
        
        tl.store(
            output_ptr + offs_m * stride_xm + offs_n_curr * stride_xn,
            output,
            mask=mask
        )
```

### 3. Batch Normalization (推理模式)

```python
@triton.jit
def batchnorm_inference_kernel(
    x_ptr, output_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    n_elements,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # 加载参数（假设已经在正确的形状）
    x = tl.load(x_ptr + offsets, mask=mask)
    mean = tl.load(mean_ptr + offsets, mask=mask)
    var = tl.load(var_ptr + offsets, mask=mask)
    weight = tl.load(weight_ptr + offsets, mask=mask)
    bias = tl.load(bias_ptr + offsets, mask=mask)
    
    # 归一化
    rstd = 1.0 / tl.sqrt(var + eps)
    normalized = (x - mean) * rstd
    output = normalized * weight + bias
    
    tl.store(output_ptr + offsets, output, mask=mask)
```

---

## 实用技巧

### 1. 处理非对齐的形状

```python
# 使用掩码处理边界
mask = offsets < n_elements
data = tl.load(ptr + offsets, mask=mask, other=0.0)
```

### 2. 多维索引

```python
# 2D 索引
row_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
col_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
ptrs = base_ptr + row_offsets[:, None] * stride_row + col_offsets[None, :] * stride_col
```

### 3. 使用 constexpr 优化

```python
# 将编译时常量标记为 tl.constexpr
BLOCK_SIZE: tl.constexpr  # 可以用作张量形状
```

### 4. 性能测试模板

```python
import triton.testing

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],
        x_vals=[2**i for i in range(10, 20)],
        line_arg='provider',
        line_vals=['triton', 'torch'],
        xlabel='Size',
        ylabel='GB/s',
        plot_name='performance',
    ))
def benchmark(size, provider):
    x = torch.randn(size, device='cuda')
    quantiles = [0.5, 0.2, 0.8]
    
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch.relu(x), quantiles=quantiles
        )
    else:  # triton
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: relu_triton(x), quantiles=quantiles
        )
    
    gbps = lambda ms: 2 * x.numel() * x.element_size() / ms * 1e-6
    return gbps(ms), gbps(max_ms), gbps(min_ms)
```

---

## 调试清单

1. ✅ 检查掩码是否正确
2. ✅ 验证步长计算
3. ✅ 确认数据类型匹配
4. ✅ 测试边界情况（小输入、非对齐）
5. ✅ 使用 `TRITON_INTERPRET=1` 调试
6. ✅ 比对 PyTorch 实现结果

---

**更新日期**：2024年12月  
**适用版本**：Triton 3.6.0

更多示例请参考官方教程：`python/tutorials/`
