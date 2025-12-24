# Triton 快速参考手册

> **一页纸了解 Triton 核心概念**

## 什么是 Triton？

**Triton** 是一个用 Python 编写高性能 GPU 内核的语言和编译器，由 OpenAI 开发。

### 核心特点

- 🚀 **高生产力**：比 CUDA 简单 5-10 倍
- ⚡ **高性能**：达到手写 CUDA 的 95-105%
- 🔧 **高灵活性**：支持自定义和融合操作
- 🌐 **可移植**：同一代码支持 NVIDIA 和 AMD GPU

---

## 快速开始

### 安装

```bash
pip install triton
```

### Hello World（向量加法）

```python
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)

def add(x, y):
    out = torch.empty_like(x)
    n = out.numel()
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, out, n, BLOCK_SIZE=1024)
    return out

# 使用
x = torch.rand(10000, device='cuda')
y = torch.rand(10000, device='cuda')
z = add(x, y)
```

---

## 核心概念速查

### SPMD 编程模型

| CUDA | Triton |
|------|--------|
| 每个线程处理一个元素 | 每个程序处理一个数据块 |
| 手动管理共享内存 | 编译器自动管理 |
| 显式同步 | 自动同步 |

### 关键语言原语

```python
# 程序控制
pid = tl.program_id(axis=0)        # 获取程序 ID
offsets = tl.arange(0, BLOCK_SIZE) # 生成索引范围

# 内存操作
data = tl.load(ptr, mask=mask)     # 加载数据
tl.store(ptr, data, mask=mask)     # 存储数据
tl.atomic_add(ptr, data)           # 原子操作

# 计算操作
result = tl.dot(a, b)              # 矩阵乘法
result = tl.sum(x, axis=0)         # 归约求和
result = tl.exp(x)                 # 元素级函数
```

### 自动调优

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
    ],
    key=['n'],
)
@triton.jit
def kernel(...):
    ...
```

---

## 与 CUDA 生态对比

### cuBLAS

| | cuBLAS | Triton |
|---|--------|--------|
| **用途** | 标准 BLAS 操作 | 自定义操作 |
| **灵活性** | 低（固定算法） | 高（完全可定制） |
| **性能** | 峰值（100%） | 接近峰值（95-105%） |
| **使用建议** | 标准矩阵乘法 | 融合操作 |

### CUTLASS

| | CUTLASS | Triton |
|---|---------|--------|
| **语言** | C++ 模板 | Python |
| **学习曲线** | 陡峭 | 平缓 |
| **开发速度** | 慢 | 快 |
| **性能** | 峰值 | 接近峰值 |
| **使用建议** | 极致优化 | 快速开发 |

### cuDNN

| | cuDNN | Triton |
|---|-------|--------|
| **覆盖范围** | DNN 标准层 | 通用内核 |
| **定制性** | 低（黑盒） | 高（白盒） |
| **使用建议** | Conv、BN 等标准层 | 自定义层和融合操作 |

---

## 典型应用场景

### ✅ 适合使用 Triton

1. **融合算子**：减少内存访问
   ```python
   # 融合 GELU(matmul(x, w) + b)
   @triton.jit
   def fused_matmul_gelu(...)
   ```

2. **自定义操作**：实现新算法
   ```python
   # Flash Attention
   # Grouped GEMM
   # Custom Normalization
   ```

3. **特殊优化**：针对特定数据分布
   ```python
   # 稀疏矩阵乘法
   # 块稀疏注意力
   ```

### ❌ 不适合使用 Triton

1. **标准操作**：已有优化库
   ```python
   # 使用 torch.matmul（底层 cuBLAS）
   C = torch.matmul(A, B)
   ```

2. **复杂控制流**：可能编译失败

3. **最后 1% 优化**：考虑 CUTLASS

---

## 性能优化技巧

### 1. 选择合适的块大小

```python
# 经验法则
BLOCK_SIZE = 128 或 256  # 一般情况
BLOCK_SIZE = 64 或 32    # 内存密集型
```

### 2. 使用自动调优

```python
@triton.autotune(configs=[...], key=['M', 'N', 'K'])
```

### 3. 内存访问优化

```python
# ✅ 合并访问
offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

# ❌ 跨步访问（慢）
offs = pid + tl.arange(0, N) * stride
```

### 4. 利用 Tensor Core

```python
# 使用 tl.dot() 进行矩阵乘法
acc = tl.dot(a, b, acc)  # 自动使用 Tensor Core
```

---

## 调试技巧

### 环境变量

```bash
# 打印生成的 IR
export MLIR_ENABLE_DUMP=1

# 使用解释器（不需要 GPU）
export TRITON_INTERPRET=1

# 保存编译结果
export TRITON_KERNEL_DUMP=1
export TRITON_DUMP_DIR=./dump
```

### Python 调试

```python
# 查看生成的 PTX
print(kernel.asm['ptx'])

# 性能测试
triton.testing.do_bench(lambda: kernel[grid](...))
```

---

## 代码框架速览

```
triton/
├── python/triton/           # Python 前端
│   ├── language/           # 语言定义 (tl.*)
│   ├── compiler/           # 编译器前端
│   ├── runtime/            # JIT 和自动调优
│   └── backends/           # 硬件后端
├── lib/                    # C++ 后端（MLIR）
│   ├── Dialect/           # MLIR 方言
│   └── Conversion/        # 方言转换
└── docs/                   # 文档
```

---

## 编译流程

```
Python 代码
   ↓
Triton IR (高层)
   ↓
TritonGPU IR (GPU 优化)
   ↓
LLVM IR
   ↓
PTX/GCN (汇编)
   ↓
二进制代码
```

---

## 学习资源

### 官方资源
- **官网**：https://triton-lang.org
- **GitHub**：https://github.com/triton-lang/triton
- **教程**：`python/tutorials/` 目录

### 推荐学习路径

1. **入门**（1-2 天）
   - 阅读 `01-vector-add.py`
   - 运行 `02-fused-softmax.py`
   - 理解 SPMD 模型

2. **进阶**（1-2 周）
   - 实现 `03-matrix-multiplication.py`
   - 学习自动调优
   - 优化内存访问

3. **高级**（持续）
   - 阅读编译器源码
   - 贡献新特性
   - 研究 MLIR 方言

### 社区资源
- **Triton Puzzles**：https://github.com/srush/Triton-Puzzles
- **会议视频**：YouTube "Triton Developer Conference"

---

## 常见问题 FAQ

### Q: Triton 会替代 CUDA 吗？
**A**: 不会。Triton 是 CUDA 的**补充**，用于简化自定义内核开发。

### Q: Triton 的性能如何？
**A**: 通常达到手写 CUDA 的 95-105%，对于某些操作甚至更好。

### Q: 学习 Triton 需要了解 CUDA 吗？
**A**: 不需要。了解基本的并行计算概念即可，但了解 CUDA 有助于理解底层。

### Q: Triton 支持 CPU 吗？
**A**: 正在开发中。目前主要支持 NVIDIA 和 AMD GPU。

### Q: 如何在生产环境使用 Triton？
**A**: 许多公司（包括 OpenAI）已在生产环境使用。建议先在非关键路径测试。

---

## 实用代码片段

### 基本模板

```python
import triton
import triton.language as tl

@triton.jit
def my_kernel(
    # 指针参数
    input_ptr, output_ptr,
    # 形状参数
    n_elements,
    # 元参数（编译时常量）
    BLOCK_SIZE: tl.constexpr,
):
    # 1. 获取程序 ID
    pid = tl.program_id(0)
    
    # 2. 计算偏移量
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # 3. 创建掩码
    mask = offsets < n_elements
    
    # 4. 加载数据
    data = tl.load(input_ptr + offsets, mask=mask)
    
    # 5. 计算
    result = data * 2.0
    
    # 6. 存储结果
    tl.store(output_ptr + offsets, result, mask=mask)
```

### 启动内核

```python
def launch_kernel(input_tensor):
    output = torch.empty_like(input_tensor)
    n = output.numel()
    
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    
    my_kernel[grid](
        input_tensor, output, n,
        BLOCK_SIZE=1024
    )
    
    return output
```

---

## 性能对比总结

| 操作 | cuBLAS/cuDNN | 手写 CUDA | Triton | 开发时间 |
|------|--------------|-----------|--------|---------|
| 矩阵乘法 | 100% | 95-100% | 95-105% | 1-2 小时 |
| Softmax | - | 100% | 100-120% | 30 分钟 |
| LayerNorm | 100% | 100% | 90-110% | 1 小时 |
| 融合算子 | - | 100% | 95-105% | 2-4 小时 |

**结论**：Triton 提供了**生产力**和**性能**的最佳平衡。

---

**版本**：Triton 3.6.0  
**更新**：2024年12月

更多详细信息，请参阅完整指南：`triton-framework-guide-zh.md`
