# Triton 中文文档集

本文档集全面介绍了 Triton 框架的代码结构、核心概念、基本用法、优势以及与 CUDA 生态系统的关系。

## 📚 文档导览

### 1. [完整框架指南](./triton-framework-guide-zh.md)
**适合：想要深入了解 Triton 的开发者**

包含内容：
- ✨ Triton 概述和历史背景
- 🏗️ 详细的代码框架结构
- 💡 核心概念深度解析（SPMD、JIT、自动调优等）
- 📝 基本用法和完整示例
- 🚀 核心优势分析
- 🔗 与 CUDA 生态（cuBLAS、CUTLASS、cuDNN 等）的关系
- ⚙️ 编译流程详解
- 📖 学习资源和参考文献

**文件大小**：约 18KB  
**阅读时间**：30-45 分钟

---

### 2. [快速参考手册](./triton-quick-reference-zh.md)
**适合：需要快速查阅的开发者**

包含内容：
- 🎯 核心特点和一页纸总结
- ⚡ 快速安装和 Hello World
- 📋 核心概念速查表
- 📊 与 CUDA 工具对比表格
- ✅ 典型应用场景清单
- 💪 性能优化技巧
- 🐛 调试技巧
- ❓ 常见问题 FAQ

**文件大小**：约 6KB  
**阅读时间**：10-15 分钟

---

### 3. [实战示例集](./triton-examples-zh.md)
**适合：通过代码学习的开发者**

包含内容：
- 🔧 基础操作（元素级运算、条件选择等）
- 🔥 融合算子（FMA、融合线性层等）
- 📉 归约操作（求和、最大值、按行归约）
- 🧮 矩阵运算（转置、批量矩阵乘法）
- 🎭 激活函数（ReLU、GELU、Sigmoid、SiLU）
- 📏 规范化层（Softmax、LayerNorm、BatchNorm）
- 🛠️ 实用技巧和调试清单

**文件大小**：约 16KB  
**浏览时间**：按需查阅

---

## 🎓 推荐学习路径

### 新手入门（1-2 天）
1. 阅读 [快速参考手册](./triton-quick-reference-zh.md) 的前半部分
2. 运行官方教程 `python/tutorials/01-vector-add.py`
3. 在 [实战示例集](./triton-examples-zh.md) 中查找感兴趣的例子
4. 尝试修改示例代码

### 深入理解（1-2 周）
1. 完整阅读 [完整框架指南](./triton-framework-guide-zh.md)
2. 学习 SPMD 编程模型和内存层次
3. 实现官方教程中的矩阵乘法
4. 使用自动调优优化性能
5. 对比 Triton 和 PyTorch 的性能

### 进阶应用（持续）
1. 阅读 Triton 源码（`python/triton/` 和 `lib/`）
2. 理解 MLIR 编译流程
3. 实现自定义融合算子
4. 贡献代码到社区
5. 研究高级优化技术

---

## 📊 文档对比

| 文档 | 深度 | 广度 | 代码量 | 适合场景 |
|------|------|------|--------|---------|
| 完整指南 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 系统学习 |
| 快速参考 | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | 快速查阅 |
| 实战示例 | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 代码实践 |

---

## 🚀 快速开始

### 安装 Triton

```bash
# 从 PyPI 安装（推荐）
pip install triton

# 从源码安装
git clone https://github.com/triton-lang/triton.git
cd triton
pip install -e .
```

### 运行第一个示例

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

# 测试
x = torch.rand(10000, device='cuda')
y = torch.rand(10000, device='cuda')
z = add(x, y)
print(f"结果正确: {torch.allclose(z, x + y)}")
```

---

## 🔗 相关资源

### 官方资源
- **官网**：https://triton-lang.org
- **GitHub**：https://github.com/triton-lang/triton
- **官方文档**：https://triton-lang.org/main/index.html
- **官方教程**：`python/tutorials/`

### 社区资源
- **Triton Puzzles**：https://github.com/srush/Triton-Puzzles
- **会议视频**：[YouTube Playlist](https://www.youtube.com/playlist?list=PLc_vA1r0qoiQqCdWFDUDqI90oY5EjfGuO)

### 学术论文
- **MAPL 2019**：[Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations](http://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf)

---

## 💡 核心要点

### Triton 是什么？
一个让你用 **Python** 写出接近手写 **CUDA** 性能的 GPU 内核框架。

### 为什么选择 Triton？
- **生产力**：开发效率提升 5-10 倍
- **性能**：达到手写 CUDA 的 95-105%
- **灵活性**：支持自定义和融合操作
- **简单**：基于块的编程模型，编译器自动优化

### 何时使用 Triton？
✅ 自定义 GPU 算子  
✅ 融合多个操作  
✅ 实现新算法  
✅ 优化特定工作负载  

❌ 标准操作（用 cuBLAS/cuDNN）  
❌ 极致的最后 1-2% 性能  
❌ CPU 密集型任务  

### 与 CUDA 生态的关系

```
应用层：PyTorch, TensorFlow, JAX
         ↓
高层库：cuDNN (标准层) | Triton (自定义算子)
         ↓
中层工具：cuBLAS (BLAS) | CUTLASS (模板)
         ↓
底层：CUDA Runtime | CUDA Driver
         ↓
硬件：NVIDIA GPU
```

**Triton 的定位**：填补标准库和原始 CUDA 之间的空白。

---

## 🎯 常见问题

### Q: 学习 Triton 需要了解 CUDA 吗？
**A**: 不需要。了解基本的并行计算概念即可。但了解 CUDA 有助于理解底层原理。

### Q: Triton 会替代 CUDA 吗？
**A**: 不会。Triton 是 CUDA 的补充，用于简化自定义内核开发。标准操作仍建议使用 cuBLAS/cuDNN。

### Q: Triton 的性能如何？
**A**: 通常达到手写 CUDA 的 95-105%。对于某些融合操作，性能甚至可能更好。

### Q: 支持哪些硬件？
**A**: 
- ✅ NVIDIA GPU (Compute Capability 8.0+)
- ✅ AMD GPU (ROCm 6.2+)
- 🚧 CPU（开发中）

### Q: 如何调试 Triton 代码？
**A**: 
1. 使用 `TRITON_INTERPRET=1` 环境变量在 CPU 上解释执行
2. 使用 `MLIR_ENABLE_DUMP=1` 查看生成的 IR
3. 对比 PyTorch 实现验证正确性
4. 使用 `triton.testing.do_bench` 测试性能

---

## 📝 文档维护

### 版本信息
- **Triton 版本**：3.6.0
- **文档版本**：1.0
- **最后更新**：2024年12月

### 贡献指南
如发现文档错误或有改进建议，请：
1. 在 GitHub 上提交 Issue
2. 或直接提交 Pull Request

### 许可证
本文档遵循 Triton 项目的许可证（见仓库根目录 LICENSE 文件）。

---

## 🙏 致谢

感谢以下资源和社区：
- OpenAI Triton 团队
- Triton 开源社区
- MLIR 项目
- CUDA 生态系统

---

**开始你的 Triton 之旅吧！** 🚀

如有问题，欢迎访问：
- GitHub Issues: https://github.com/triton-lang/triton/issues
- 官方文档: https://triton-lang.org
