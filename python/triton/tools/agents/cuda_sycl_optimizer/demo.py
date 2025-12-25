#!/usr/bin/env python3
"""
Simple standalone example of the CUDA-to-SYCL Optimization Agent

This script demonstrates the agent without requiring full Triton installation.
Run from the cuda_sycl_optimizer directory.
"""

import asyncio
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from detector import ChangeDetector
from analyzer import OptimizationAnalyzer
from recommender import SyclRecommender
from pr_generator import PRGenerator, PRConfig


async def demo():
    """
    Simple demonstration of the CUDA-to-SYCL agent workflow
    """
    print("=" * 70)
    print("CUDA-to-SYCL Optimization Agent - Standalone Demo")
    print("=" * 70)
    
    # Sample CUDA code transformation
    print("\n📝 Step 1: Sample CUDA Optimization")
    print("-" * 70)
    
    old_cuda = """
__global__ void matmul(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
"""
    
    new_cuda = """
#define TILE_SIZE 16

__global__ void matmul(float* A, float* B, float* C, int N) {
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;
    
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; t++) {
        if (row < N && t * TILE_SIZE + threadIdx.x < N)
            tile_A[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_SIZE + threadIdx.x];
        else
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;
            
        if (col < N && t * TILE_SIZE + threadIdx.y < N)
            tile_B[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        else
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;
            
        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
        }
        __syncthreads();
    }
    
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}
"""
    
    print("BEFORE (naive implementation):")
    print(old_cuda[:200] + "...")
    print("\nAFTER (tiled with shared memory):")
    print(new_cuda[:300] + "...")
    
    # Analyze the optimization
    print("\n🔍 Step 2: Analyzing CUDA Optimization")
    print("-" * 70)
    
    analyzer = OptimizationAnalyzer()
    analysis = await analyzer.analyze_change(
        change_id="demo/matmul-optimization",
        old_code=old_cuda,
        new_code=new_cuda,
        metadata={
            "title": "Optimize matmul with shared memory tiling",
            "description": "Use tiled matrix multiplication with shared memory"
        }
    )
    
    print(f"✅ Analysis complete!")
    print(f"   Patterns detected: {len(analysis.patterns)}")
    print(f"   SYCL applicability: {analysis.sycl_applicability:.1%}")
    print(f"   Implementation complexity: {analysis.complexity}")
    
    if analysis.performance_impact:
        print(f"   Expected speedup: {analysis.performance_impact.expected_speedup}")
    
    print(f"\n   Detected optimization patterns:")
    for i, pattern in enumerate(analysis.patterns, 1):
        print(f"   {i}. {pattern.pattern_type.value} (confidence: {pattern.confidence:.1%})")
        print(f"      → {pattern.description}")
    
    # Generate SYCL recommendations
    print("\n💡 Step 3: Generating SYCL Recommendations")
    print("-" * 70)
    
    recommender = SyclRecommender()
    recommendations = await recommender.generate_recommendations(
        analysis_report=analysis,
        target_repo="example/sycl-kernels"
    )
    
    print(f"✅ Generated {len(recommendations.recommendations)} recommendation(s)")
    print(f"   Overall priority: {recommendations.overall_priority.upper()}")
    
    for i, rec in enumerate(recommendations.recommendations, 1):
        print(f"\n   Recommendation {i}: {rec.title}")
        print(f"   ├─ Priority: {rec.priority}")
        print(f"   ├─ Expected impact: {rec.expected_impact}")
        print(f"   ├─ Effort: {rec.implementation_effort}")
        print(f"   └─ Risk: {rec.risk_level}")
    
    # Show SYCL code example
    if recommendations.recommendations:
        print("\n📄 Step 4: Sample SYCL Implementation")
        print("-" * 70)
        rec = recommendations.recommendations[0]
        print(f"For: {rec.title}")
        print("\nGenerated SYCL code:")
        print(rec.sycl_code[:500] + "...\n")
    
    # Generate PR preview
    print("📦 Step 5: Pull Request Preview")
    print("-" * 70)
    
    pr_gen = PRGenerator(github_token="demo-token")
    config = PRConfig(
        target_owner="example",
        target_repo="sycl-kernels",
        base_branch="main",
        draft_mode=True,
        labels=["optimization", "sycl", "auto-generated"]
    )
    
    pr = await pr_gen.create_pr(
        recommendation_report=recommendations,
        config=config
    )
    
    print(f"✅ PR generated!")
    print(f"   Branch: {pr.branch_name}")
    print(f"   Title: {pr.title}")
    print(f"   Status: {pr.status}")
    print(f"   Files to be changed: {len(pr.files_changed)}")
    
    print("\n   PR Description (preview):")
    print("   " + "-" * 66)
    for line in pr.body.split('\n')[:20]:
        print(f"   {line}")
    print("   ...")
    print("   " + "-" * 66)
    
    # Summary
    print("\n" + "=" * 70)
    print("Demo Summary")
    print("=" * 70)
    print(f"✅ Detected {len(analysis.patterns)} optimization patterns in CUDA code")
    print(f"✅ Generated {len(recommendations.recommendations)} SYCL recommendations")
    print(f"✅ Created PR with {len(pr.files_changed)} file changes")
    print(f"\nThe agent successfully:")
    print("  1. Analyzed CUDA kernel optimizations")
    print("  2. Identified shared memory tiling and loop unrolling patterns")
    print("  3. Mapped patterns to SYCL equivalents")
    print("  4. Generated implementation code and PR description")
    print("\n💡 This demonstrates the full CUDA → SYCL optimization workflow!")
    print("=" * 70)


if __name__ == "__main__":
    print("\n" + "🚀 Starting CUDA-to-SYCL Agent Demo...\n")
    
    try:
        asyncio.run(demo())
        print("\n✅ Demo completed successfully!\n")
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user\n")
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        import traceback
        traceback.print_exc()
