# CUDA-to-SYCL Optimization Agent - Quick Start Guide

## What is This?

The CUDA-to-SYCL Optimization Agent is an intelligent system that automatically:
1. **Monitors** PyTorch and CUDA repositories for kernel optimizations
2. **Analyzes** the optimization patterns used in CUDA code
3. **Translates** those optimizations to SYCL equivalents
4. **Generates** pull requests with SYCL implementations

This helps teams maintaining SYCL kernels stay synchronized with CUDA optimizations without manual tracking and translation.

## Quick Demo

Run the standalone demo to see the agent in action:

```bash
cd python/triton/tools/agents/cuda_sycl_optimizer
python demo.py
```

This will demonstrate:
- CUDA optimization analysis (shared memory tiling example)
- Pattern detection (identifies shared memory, loop unrolling, etc.)
- SYCL code generation with proper mappings
- PR description generation

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    CUDA-SYCL Agent                       │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  Detector → Analyzer → Recommender → PR Generator       │
│     ↓           ↓            ↓              ↓            │
│  GitHub     Pattern       SYCL          Auto PR         │
│  Monitor    Matching    Templates      Creation         │
│                                                           │
└─────────────────────────────────────────────────────────┘
```

## Components

### 1. Detector Module (`detector.py`)
Monitors GitHub for CUDA kernel changes:
- Watches specified repositories (PyTorch, CUDA samples, etc.)
- Filters by file patterns (`*.cu`, `*.cuh`)
- Classifies by type (optimization, new kernel, bug fix)
- Prioritizes changes

### 2. Analyzer Module (`analyzer.py`)
Identifies optimization patterns:
- **Memory**: Coalescing, shared memory, bank conflicts
- **Compute**: Loop unrolling, register optimization
- **Sync**: Barrier reduction, warp shuffles
- **Arch**: Tensor cores, async copies

Outputs confidence scores and SYCL applicability ratings.

### 3. Recommender Module (`recommender.py`)
Generates SYCL recommendations:
- Maps CUDA patterns to SYCL equivalents
- Provides code templates
- Ranks by priority
- Estimates effort and risk

### 4. PR Generator Module (`pr_generator.py`)
Creates pull requests:
- Generates SYCL implementations
- Creates test and benchmark code
- Writes comprehensive PR descriptions
- Links to original CUDA changes

## Key Features

### CUDA → SYCL Translation Examples

| CUDA | SYCL |
|------|------|
| `__shared__` | `sycl::local_accessor` |
| `__syncthreads()` | `item.barrier()` |
| `threadIdx.x` | `item.get_local_id(0)` |
| `blockIdx.x` | `item.get_group_id(0)` |
| `__shfl_down()` | `sycl::shift_group_left()` |
| `wmma` (tensor cores) | `joint_matrix` |

### Supported Optimization Patterns

1. **Memory Coalescing** - Sequential access patterns
2. **Shared Memory** - Local memory for data reuse
3. **Loop Unrolling** - Instruction-level parallelism
4. **Warp Shuffle** - Sub-group communication
5. **Tensor Cores** - Matrix operations
6. **Async Copy** - Asynchronous transfers
7. **Barrier Reduction** - Minimize synchronization
8. **Register Optimization** - Register blocking
9. **Bank Conflict Avoidance** - Memory padding
10. **Occupancy Tuning** - Thread configuration

## Usage Examples

### Example 1: Basic Analysis

```python
import asyncio
from analyzer import OptimizationAnalyzer

async def analyze():
    analyzer = OptimizationAnalyzer()
    
    report = await analyzer.analyze_change(
        change_id="pytorch/pytorch#12345",
        old_code=old_cuda_code,
        new_code=new_cuda_code
    )
    
    print(f"Patterns: {len(report.patterns)}")
    print(f"SYCL applicability: {report.sycl_applicability:.0%}")
    print(f"Complexity: {report.complexity}")

asyncio.run(analyze())
```

### Example 2: Generate Recommendations

```python
from recommender import SyclRecommender

async def recommend():
    recommender = SyclRecommender()
    
    recommendations = await recommender.generate_recommendations(
        analysis_report=report,
        target_repo="my-org/sycl-kernels"
    )
    
    for rec in recommendations.recommendations:
        print(f"{rec.title}: {rec.expected_impact}")

asyncio.run(recommend())
```

### Example 3: Monitor Repositories

```python
from detector import ChangeDetector

async def monitor():
    detector = ChangeDetector(api_token="your_token")
    
    detector.add_repository(
        owner="pytorch",
        repo="pytorch",
        watch_patterns=["*.cu", "*.cuh"]
    )
    
    await detector.start_monitoring()

asyncio.run(monitor())
```

## Configuration

Edit `config.yaml` to customize:

```yaml
detection:
  repositories:
    - owner: "pytorch"
      repo: "pytorch"
      branches: ["main"]
      paths: ["aten/src/ATen/cuda/**"]
  
  filters:
    include_labels: ["performance", "optimization"]
    exclude_labels: ["documentation"]

recommendation:
  target_repository:
    owner: "my-org"
    repo: "sycl-kernels"
    
pr_generation:
  auto_create: false
  draft_mode: true
  labels: ["optimization", "auto-generated"]
```

## Testing the Agent

### Run the Demo
```bash
python demo.py
```

### Test Individual Modules
```bash
# Test analyzer
python analyzer.py

# Test recommender  
python recommender.py

# Test PR generator
python pr_generator.py
```

### Import in Your Code
```python
# As a package (requires Triton installed)
from triton.tools.agents.cuda_sycl_optimizer import (
    ChangeDetector,
    OptimizationAnalyzer,
    SyclRecommender,
    PRGenerator
)

# Standalone (from this directory)
from detector import ChangeDetector
from analyzer import OptimizationAnalyzer
from recommender import SyclRecommender
from pr_generator import PRGenerator
```

## Real-World Workflow

1. **Setup Monitoring**
   ```bash
   # Configure repositories in config.yaml
   # Set GITHUB_TOKEN environment variable
   python -m cuda_sycl_optimizer.detector
   ```

2. **Automatic Detection**
   - Agent polls repositories every hour
   - Detects CUDA kernel PRs with optimization labels
   - Filters by file patterns (*.cu, *.cuh)

3. **Analysis & Recommendation**
   - Analyzes code diffs automatically
   - Identifies optimization patterns
   - Generates SYCL recommendations
   - Ranks by priority and impact

4. **PR Creation**
   - Generates SYCL implementation code
   - Creates comprehensive PR description
   - Links to original CUDA PR
   - Optionally auto-creates PR (if configured)

5. **Review & Merge**
   - Team reviews generated PR
   - Runs tests and benchmarks
   - Merges if approved

## Output Examples

### Analysis Report
```
Patterns detected: 3
- memory_coalescing (confidence: 85%)
- shared_memory (confidence: 90%)
- loop_unrolling (confidence: 88%)

SYCL applicability: 87%
Expected speedup: 20-30%
Complexity: medium
```

### Generated SYCL Code
```cpp
queue.submit([&](sycl::handler& cgh) {
    auto local_acc = sycl::local_accessor<float, 1>(
        sycl::range<1>(256), cgh);
    
    cgh.parallel_for(sycl::nd_range<1>(global, local),
        [=](sycl::nd_item<1> item) {
        size_t lid = item.get_local_id(0);
        local_acc[lid] = input_acc[item.get_global_id(0)];
        item.barrier();
        // ... use local memory
    });
});
```

### PR Description
```markdown
## CUDA Optimization Adaptation

### Source
- CUDA PR: pytorch/pytorch#12345
- Optimization: Shared Memory Tiling
- Performance Gain: 20-30%

### Changes
This PR adapts CUDA memory optimizations to SYCL...

### SYCL Adaptations
- Replaced `__shared__` with local accessor
- Mapped thread blocks to work-groups
- Used sub-group operations for warp-level code

### Testing
- [x] Unit tests pass
- [x] Performance benchmarks show 25% improvement
- [x] Correctness validation complete
```

## Limitations & Future Work

### Current Limitations
- Requires manual review of generated code
- Performance estimates are approximate
- May not handle all edge cases
- Limited to common patterns

### Planned Enhancements
- ML-based pattern recognition
- Multi-target support (HIP, Metal)
- Automated performance validation
- Bidirectional SYCL → CUDA sync
- Web dashboard for monitoring
- IDE integration

## Documentation

- **Full Design**: `docs/agents/cuda-to-sycl-optimization-agent.md`
- **Module README**: `README.md`
- **API Reference**: See docstrings in each module
- **Configuration**: `config.yaml` with comments

## Support & Contributing

- Report issues on GitHub
- See main Triton CONTRIBUTING.md for guidelines
- Add new patterns by extending analyzer and recommender
- Improve translation rules in recommender module

## Related Projects

- **Triton**: https://triton-lang.org
- **TritonForge**: https://github.com/RLsys-Foundation/TritonForge
- **SYCL**: https://www.khronos.org/sycl/
- **PyTorch**: https://github.com/pytorch/pytorch

## License

See main Triton LICENSE file.

---

**Quick Start**: `python demo.py`

**Full Documentation**: See `docs/agents/cuda-to-sycl-optimization-agent.md`
