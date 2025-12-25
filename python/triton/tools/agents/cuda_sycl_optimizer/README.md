# CUDA-to-SYCL Optimization Agent

An intelligent agent system for automatically detecting PyTorch CUDA kernel updates and providing SYCL optimization recommendations.

## Overview

This agent monitors CUDA kernel changes in repositories like PyTorch and NVIDIA CUDA samples, analyzes optimization patterns, and generates actionable SYCL recommendations with automated PR generation capabilities.

## Features

- **Automatic Detection**: Monitors GitHub repositories for CUDA kernel changes
- **Intelligent Analysis**: Identifies optimization patterns using static analysis and ML
- **SYCL Translation**: Maps CUDA optimizations to SYCL equivalents
- **PR Generation**: Automatically creates pull requests with optimized SYCL code
- **Performance Estimation**: Predicts performance impact of optimizations
- **Priority Ranking**: Ranks recommendations by impact and feasibility

## Architecture

### Core Modules

1. **Change Detector** (`detector.py`)
   - Monitors GitHub repositories via API
   - Filters relevant CUDA kernel changes
   - Classifies changes by type and priority

2. **Optimization Analyzer** (`analyzer.py`)
   - Analyzes code diffs for optimization patterns
   - Identifies memory, compute, and synchronization optimizations
   - Estimates performance impact

3. **SYCL Recommender** (`recommender.py`)
   - Translates CUDA patterns to SYCL equivalents
   - Generates code templates
   - Ranks recommendations by priority

4. **PR Generator** (`pr_generator.py`)
   - Creates feature branches
   - Generates SYCL implementations
   - Creates comprehensive pull requests

## Installation

```bash
# Install from triton repository
pip install -e /path/to/triton

# Or install dependencies manually
pip install asyncio aiohttp PyGithub
```

## Quick Start

### 1. Basic Usage

```python
import asyncio
from triton.tools.agents.cuda_sycl_optimizer import (
    ChangeDetector,
    OptimizationAnalyzer,
    SyclRecommender,
    PRGenerator
)

async def main():
    # Initialize components
    detector = ChangeDetector(api_token="your_github_token")
    analyzer = OptimizationAnalyzer()
    recommender = SyclRecommender()
    
    # Add repositories to monitor
    detector.add_repository(
        owner="pytorch",
        repo="pytorch",
        branches=["main"],
        watch_patterns=["*.cu", "*.cuh"]
    )
    
    # Start monitoring
    await detector.start_monitoring()

asyncio.run(main())
```

### 2. Analyze a Specific Change

```python
async def analyze_change():
    analyzer = OptimizationAnalyzer()
    
    # Analyze code diff
    report = await analyzer.analyze_change(
        change_id="pytorch/pytorch#12345",
        old_code=old_cuda_code,
        new_code=new_cuda_code
    )
    
    print(f"Patterns found: {len(report.patterns)}")
    print(f"SYCL applicability: {report.sycl_applicability:.2f}")
    print(f"Summary: {report.summary}")

asyncio.run(analyze_change())
```

### 3. Generate SYCL Recommendations

```python
async def generate_recommendations():
    analyzer = OptimizationAnalyzer()
    recommender = SyclRecommender()
    
    # Analyze CUDA changes
    analysis = await analyzer.analyze_change(...)
    
    # Generate SYCL recommendations
    recommendations = await recommender.generate_recommendations(
        analysis_report=analysis,
        target_repo="my-org/sycl-kernels"
    )
    
    for rec in recommendations.recommendations:
        print(f"{rec.title}: {rec.expected_impact}")

asyncio.run(generate_recommendations())
```

### 4. Create a Pull Request

```python
async def create_optimization_pr():
    recommender = SyclRecommender()
    pr_gen = PRGenerator(github_token="your_token")
    
    # Generate recommendations
    recommendations = await recommender.generate_recommendations(...)
    
    # Create PR
    config = PRConfig(
        target_owner="my-org",
        target_repo="sycl-kernels",
        base_branch="main",
        draft_mode=True
    )
    
    pr = await pr_gen.create_pr(
        recommendation_report=recommendations,
        config=config
    )
    
    print(f"Created PR: {pr.url}")

asyncio.run(create_optimization_pr())
```

## Configuration

### Repository Configuration

```python
detector.add_repository(
    owner="pytorch",
    repo="pytorch",
    branches=["main", "release"],
    watch_patterns=["*.cu", "*.cuh", "aten/src/ATen/cuda/**"],
    include_labels={"performance", "optimization", "cuda"},
    exclude_labels={"documentation", "wip"}
)
```

### PR Configuration

```python
config = PRConfig(
    target_owner="my-org",
    target_repo="sycl-kernels",
    base_branch="main",
    draft_mode=True,
    assign_reviewers=["reviewer1", "reviewer2"],
    labels=["optimization", "auto-generated", "needs-review"]
)
```

## Supported Optimization Patterns

### Memory Optimizations
- **Memory Coalescing**: Sequential memory access patterns
- **Shared Memory**: Local memory usage for data reuse
- **Bank Conflict Avoidance**: Padding to avoid conflicts

### Compute Optimizations
- **Loop Unrolling**: Instruction-level parallelism
- **Register Optimization**: Register blocking and reuse
- **Occupancy Tuning**: Thread/block configuration

### Synchronization Optimizations
- **Barrier Reduction**: Minimize synchronization points
- **Warp Shuffle**: Intra-warp communication
- **Cooperative Groups**: Advanced synchronization

### Architecture-Specific
- **Tensor Cores**: Matrix operations via joint_matrix
- **Async Copy**: Asynchronous memory transfers

## CUDA to SYCL Translation Guide

### Thread Hierarchy

| CUDA | SYCL |
|------|------|
| thread | work-item |
| block | work-group |
| grid | nd-range |
| warp | sub-group |

### Memory Types

| CUDA | SYCL |
|------|------|
| `__shared__` | `local_accessor` |
| `__constant__` | read-only accessor |
| `__global__` | global memory (default) |

### Synchronization

| CUDA | SYCL |
|------|------|
| `__syncthreads()` | `item.barrier()` |
| `__syncwarp()` | `sg.barrier()` |

### Indexing

| CUDA | SYCL |
|------|------|
| `threadIdx.x` | `item.get_local_id(0)` |
| `blockIdx.x` | `item.get_group_id(0)` |
| `blockDim.x` | `item.get_local_range(0)` |
| `gridDim.x` | `item.get_group_range(0)` |

## Examples

### Example 1: Memory Coalescing

**CUDA (Before)**:
```cuda
__global__ void kernel(float* data) {
    int idx = threadIdx.x * 128 + blockIdx.x;
    data[idx] = data[idx] * 2.0f;
}
```

**CUDA (After)**:
```cuda
__global__ void kernel(float* data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    data[idx] = data[idx] * 2.0f;
}
```

**SYCL (Generated)**:
```cpp
queue.submit([&](sycl::handler& cgh) {
    auto data_acc = data_buf.get_access<sycl::access::mode::read_write>(cgh);
    
    cgh.parallel_for(sycl::nd_range<1>(global_size, local_size),
        [=](sycl::nd_item<1> item) {
        size_t idx = item.get_global_id(0);
        data_acc[idx] = data_acc[idx] * 2.0f;
    });
});
```

### Example 2: Shared Memory

**CUDA**:
```cuda
__global__ void kernel(float* input, float* output) {
    __shared__ float smem[256];
    int lid = threadIdx.x;
    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    
    smem[lid] = input[gid];
    __syncthreads();
    
    output[gid] = smem[lid] * 2.0f;
}
```

**SYCL (Generated)**:
```cpp
queue.submit([&](sycl::handler& cgh) {
    auto input_acc = input_buf.get_access<sycl::access::mode::read>(cgh);
    auto output_acc = output_buf.get_access<sycl::access::mode::write>(cgh);
    auto local_acc = sycl::local_accessor<float, 1>(sycl::range<1>(256), cgh);
    
    cgh.parallel_for(sycl::nd_range<1>(global_size, local_size),
        [=](sycl::nd_item<1> item) {
        size_t lid = item.get_local_id(0);
        size_t gid = item.get_global_id(0);
        
        local_acc[lid] = input_acc[gid];
        item.barrier(sycl::access::fence_space::local_space);
        
        output_acc[gid] = local_acc[lid] * 2.0f;
    });
});
```

## API Reference

### ChangeDetector

```python
class ChangeDetector:
    def __init__(self, api_token: str, poll_interval: int = 3600)
    def add_repository(self, owner: str, repo: str, ...)
    async def start_monitoring(self)
    def stop_monitoring(self)
    async def fetch_change_details(self, change_id: str) -> KernelChange
```

### OptimizationAnalyzer

```python
class OptimizationAnalyzer:
    def __init__(self)
    async def analyze_change(
        self,
        change_id: str,
        old_code: Optional[str],
        new_code: Optional[str],
        metadata: Optional[Dict]
    ) -> OptimizationReport
```

### SyclRecommender

```python
class SyclRecommender:
    def __init__(self)
    async def generate_recommendations(
        self,
        analysis_report: OptimizationReport,
        target_repo: str
    ) -> RecommendationReport
```

### PRGenerator

```python
class PRGenerator:
    def __init__(self, github_token: str)
    async def create_pr(
        self,
        recommendation_report: RecommendationReport,
        config: PRConfig,
        selected_recommendations: Optional[List[str]]
    ) -> GeneratedPR
```

## Testing

```bash
# Run tests
python -m pytest python/triton/tools/agents/cuda_sycl_optimizer/tests/

# Run specific test
python -m pytest python/triton/tools/agents/cuda_sycl_optimizer/tests/test_analyzer.py
```

## Contributing

Contributions are welcome! Please see the main Triton [CONTRIBUTING.md](../../../../../CONTRIBUTING.md) for guidelines.

### Adding New Patterns

To add a new optimization pattern:

1. Add pattern type to `OptimizationPattern` enum in `analyzer.py`
2. Implement detection logic in `OptimizationAnalyzer`
3. Add SYCL template in `SyclRecommender._load_sycl_templates()`
4. Update translation rules if needed

## Limitations

- Currently supports common optimization patterns
- Requires manual review of generated code
- Performance estimates are approximate
- May not handle all edge cases

## Roadmap

- [ ] Enhanced pattern recognition using ML models
- [ ] Support for more SYCL implementations (DPC++, hipSYCL, ComputeCpp)
- [ ] Automated performance validation
- [ ] Integration with CI/CD pipelines
- [ ] Web dashboard for monitoring and approval
- [ ] Support for bidirectional SYCL → CUDA optimization sharing

## References

- [Triton Documentation](https://triton-lang.org)
- [SYCL Specification](https://www.khronos.org/sycl/)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [TritonForge](https://github.com/RLsys-Foundation/TritonForge)

## License

See the main Triton [LICENSE](../../../../../LICENSE) file.

## Support

For issues and questions:
- Open an issue on [GitHub](https://github.com/triton-lang/triton/issues)
- Refer to the [comprehensive design document](../../../../docs/agents/cuda-to-sycl-optimization-agent.md)
