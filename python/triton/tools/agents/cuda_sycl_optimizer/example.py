#!/usr/bin/env python3
"""
Example usage of the CUDA-to-SYCL Optimization Agent

This script demonstrates various ways to use the agent system.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from triton.tools.agents.cuda_sycl_optimizer import (
    ChangeDetector,
    OptimizationAnalyzer,
    SyclRecommender,
    PRGenerator
)
from triton.tools.agents.cuda_sycl_optimizer.pr_generator import PRConfig


async def example_1_monitor_repositories():
    """
    Example 1: Monitor repositories for CUDA changes
    """
    print("=" * 60)
    print("Example 1: Monitor Repositories")
    print("=" * 60)
    
    # Get GitHub token from environment
    github_token = os.getenv("GITHUB_TOKEN", "your_token_here")
    
    # Initialize detector
    detector = ChangeDetector(api_token=github_token, poll_interval=3600)
    
    # Add repositories to monitor
    detector.add_repository(
        owner="pytorch",
        repo="pytorch",
        branches=["main"],
        watch_patterns=["*.cu", "*.cuh", "aten/src/ATen/cuda/**"],
    )
    
    detector.add_repository(
        owner="NVIDIA",
        repo="cuda-samples",
        branches=["master"],
        watch_patterns=["Samples/**/*.cu", "Samples/**/*.cuh"],
    )
    
    print("\nConfigured repositories:")
    for repo in detector.repositories:
        print(f"  - {repo.owner}/{repo.repo} (branches: {repo.branches})")
    
    print("\nNote: In production, call await detector.start_monitoring()")
    print("This will continuously monitor repositories for changes.")


async def example_2_analyze_cuda_optimization():
    """
    Example 2: Analyze a specific CUDA optimization
    """
    print("\n" + "=" * 60)
    print("Example 2: Analyze CUDA Optimization")
    print("=" * 60)
    
    # Sample CUDA code before optimization
    old_code = """
__global__ void matmul_kernel(float* A, float* B, float* C, int N) {
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
    
    # Sample CUDA code after optimization (with shared memory)
    new_code = """
#define TILE_SIZE 16

__global__ void matmul_kernel(float* A, float* B, float* C, int N) {
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tiles into shared memory
        if (row < N && t * TILE_SIZE + threadIdx.x < N)
            tile_A[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_SIZE + threadIdx.x];
        else
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;
            
        if (col < N && t * TILE_SIZE + threadIdx.y < N)
            tile_B[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        else
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;
            
        __syncthreads();
        
        // Compute partial sum
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
    
    # Initialize analyzer
    analyzer = OptimizationAnalyzer()
    
    print("\nAnalyzing CUDA kernel optimization...")
    
    # Analyze the change
    report = await analyzer.analyze_change(
        change_id="pytorch/pytorch#12345",
        old_code=old_code,
        new_code=new_code,
        metadata={
            "title": "Optimize matmul kernel with tiling",
            "description": "Add shared memory tiling for better performance"
        }
    )
    
    print(f"\n📊 Analysis Results:")
    print(f"   Patterns found: {len(report.patterns)}")
    print(f"   SYCL applicability: {report.sycl_applicability:.2%}")
    print(f"   Complexity: {report.complexity}")
    
    if report.performance_impact:
        print(f"   Expected speedup: {report.performance_impact.expected_speedup}")
        print(f"   Confidence: {report.performance_impact.confidence:.2%}")
    
    print(f"\n📝 Summary:")
    print(f"   {report.summary}")
    
    print(f"\n🔍 Detected Patterns:")
    for i, pattern in enumerate(report.patterns, 1):
        print(f"   {i}. {pattern.pattern_type.value}")
        print(f"      Confidence: {pattern.confidence:.2%}")
        print(f"      Description: {pattern.description}")
    
    print(f"\n💡 Recommendations:")
    for i, rec in enumerate(report.recommendations, 1):
        print(f"   {i}. {rec}")
    
    return report


async def example_3_generate_sycl_recommendations(analysis_report):
    """
    Example 3: Generate SYCL recommendations from analysis
    """
    print("\n" + "=" * 60)
    print("Example 3: Generate SYCL Recommendations")
    print("=" * 60)
    
    # Initialize recommender
    recommender = SyclRecommender()
    
    print("\nGenerating SYCL recommendations...")
    
    # Generate recommendations
    rec_report = await recommender.generate_recommendations(
        analysis_report=analysis_report,
        target_repo="my-org/sycl-kernels"
    )
    
    print(f"\n📦 Recommendation Report:")
    print(f"   CUDA Change: {rec_report.cuda_change_id}")
    print(f"   Overall Priority: {rec_report.overall_priority}")
    print(f"   Number of Recommendations: {len(rec_report.recommendations)}")
    
    print(f"\n📝 Summary:")
    print(f"   {rec_report.summary}")
    
    print(f"\n🎯 Recommendations:")
    for i, rec in enumerate(rec_report.recommendations, 1):
        print(f"\n   {i}. {rec.title}")
        print(f"      Priority: {rec.priority.upper()}")
        print(f"      Pattern: {rec.pattern.value}")
        print(f"      Expected Impact: {rec.expected_impact}")
        print(f"      Implementation Effort: {rec.implementation_effort}")
        print(f"      Risk Level: {rec.risk_level}")
        print(f"      Description: {rec.description}")
        
        if rec.affected_files:
            print(f"      Affected Files:")
            for file in rec.affected_files:
                print(f"        - {file}")
    
    return rec_report


async def example_4_generate_pr(rec_report):
    """
    Example 4: Generate a pull request with SYCL optimizations
    """
    print("\n" + "=" * 60)
    print("Example 4: Generate Pull Request")
    print("=" * 60)
    
    # Get GitHub token from environment
    github_token = os.getenv("GITHUB_TOKEN", "your_token_here")
    
    # Initialize PR generator
    pr_gen = PRGenerator(github_token=github_token)
    
    # Configure PR settings
    config = PRConfig(
        target_owner="my-org",
        target_repo="sycl-kernels",
        base_branch="main",
        draft_mode=True,
        assign_reviewers=["reviewer1", "reviewer2"],
        labels=["optimization", "auto-generated", "sycl"]
    )
    
    print("\nGenerating pull request...")
    
    # Create PR
    pr = await pr_gen.create_pr(
        recommendation_report=rec_report,
        config=config
    )
    
    print(f"\n✅ Pull Request Generated:")
    print(f"   Branch: {pr.branch_name}")
    print(f"   Title: {pr.title}")
    print(f"   Status: {pr.status}")
    print(f"   Files Changed: {len(pr.files_changed)}")
    
    print(f"\n📄 PR Description Preview:")
    print("-" * 60)
    # Print first 500 characters of PR body
    print(pr.body[:500] + "...")
    print("-" * 60)
    
    if pr.url:
        print(f"\n🔗 PR URL: {pr.url}")
    else:
        print("\n⚠️  PR not created (dry run mode)")
    
    return pr


async def example_5_generate_test_code(rec_report):
    """
    Example 5: Generate test and benchmark code
    """
    print("\n" + "=" * 60)
    print("Example 5: Generate Test and Benchmark Code")
    print("=" * 60)
    
    if not rec_report.recommendations:
        print("No recommendations available for testing")
        return
    
    github_token = os.getenv("GITHUB_TOKEN", "your_token_here")
    pr_gen = PRGenerator(github_token=github_token)
    
    # Get first recommendation
    rec = rec_report.recommendations[0]
    
    print(f"\nGenerating test code for: {rec.title}")
    
    # Generate test code
    test_code = await pr_gen.generate_test_code(rec)
    
    print("\n📝 Test Code Preview:")
    print("-" * 60)
    print(test_code[:800] + "...")
    print("-" * 60)
    
    # Generate benchmark code
    benchmark_code = await pr_gen.generate_benchmark(rec)
    
    print("\n📊 Benchmark Code Preview:")
    print("-" * 60)
    print(benchmark_code[:800] + "...")
    print("-" * 60)


async def main():
    """
    Main function running all examples
    """
    print("\n" + "=" * 60)
    print("CUDA-to-SYCL Optimization Agent - Examples")
    print("=" * 60)
    
    # Example 1: Setup monitoring (no actual monitoring for demo)
    await example_1_monitor_repositories()
    
    # Example 2: Analyze a CUDA optimization
    analysis_report = await example_2_analyze_cuda_optimization()
    
    # Example 3: Generate SYCL recommendations
    rec_report = await example_3_generate_sycl_recommendations(analysis_report)
    
    # Example 4: Generate a pull request
    await example_4_generate_pr(rec_report)
    
    # Example 5: Generate test and benchmark code
    await example_5_generate_test_code(rec_report)
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)
    print("\nNote: These examples use mock/demo data.")
    print("In production, set GITHUB_TOKEN environment variable")
    print("and configure the agent with config.yaml")
    print("\nFor more information, see:")
    print("  - README.md")
    print("  - docs/agents/cuda-to-sycl-optimization-agent.md")


if __name__ == "__main__":
    # Check Python version
    if sys.version_info < (3, 10):
        print("Warning: This agent requires Python 3.10+")
        print(f"Current version: {sys.version}")
    
    # Run examples
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
