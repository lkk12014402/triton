"""
SYCL Recommender Module

Generates SYCL-specific optimization recommendations based on CUDA analysis.
"""

import logging
from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum

from .analyzer import OptimizationPattern, OptimizationReport

logger = logging.getLogger(__name__)


@dataclass
class SyclCodeTemplate:
    """Template for SYCL code generation"""
    pattern_type: OptimizationPattern
    template_code: str
    includes: List[str]
    notes: str


@dataclass
class SyclRecommendation:
    """A single SYCL optimization recommendation"""
    id: str
    priority: str  # "critical", "high", "medium", "low"
    pattern: OptimizationPattern
    title: str
    description: str
    sycl_code: str
    affected_files: List[str]
    expected_impact: str
    implementation_effort: str
    risk_level: str
    test_strategy: str
    additional_notes: str


@dataclass
class RecommendationReport:
    """Complete recommendation report for SYCL implementation"""
    cuda_change_id: str
    recommendations: List[SyclRecommendation]
    overall_priority: str
    summary: str
    links: Dict[str, str]


class SyclRecommender:
    """
    Generates SYCL-specific optimization recommendations.
    
    This class translates CUDA optimization patterns into SYCL equivalents
    and generates actionable recommendations with code examples.
    
    Example:
        >>> recommender = SyclRecommender()
        >>> report = await recommender.generate_recommendations(analysis_report)
        >>> for rec in report.recommendations:
        ...     print(f"{rec.title}: {rec.expected_impact}")
    """
    
    def __init__(self):
        """Initialize recommender with SYCL templates."""
        self.templates = self._load_sycl_templates()
        self.translation_rules = self._load_translation_rules()
        
    def _load_sycl_templates(self) -> Dict[OptimizationPattern, SyclCodeTemplate]:
        """
        Load SYCL code templates for different optimization patterns.
        
        Returns:
            Dictionary mapping patterns to SYCL templates
        """
        return {
            OptimizationPattern.MEMORY_COALESCING: SyclCodeTemplate(
                pattern_type=OptimizationPattern.MEMORY_COALESCING,
                template_code="""
// SYCL coalesced memory access
queue.submit([&](sycl::handler& cgh) {
    auto input_acc = input_buf.get_access<sycl::access::mode::read>(cgh);
    auto output_acc = output_buf.get_access<sycl::access::mode::write>(cgh);
    
    cgh.parallel_for(sycl::nd_range<1>(global_size, local_size), 
        [=](sycl::nd_item<1> item) {
        // Ensure coalesced access: sequential work-items access sequential memory
        size_t gid = item.get_global_id(0);
        output_acc[gid] = input_acc[gid] * 2.0f;
    });
});
""",
                includes=["<sycl/sycl.hpp>"],
                notes="Work-items in the same work-group should access contiguous memory"
            ),
            
            OptimizationPattern.SHARED_MEMORY: SyclCodeTemplate(
                pattern_type=OptimizationPattern.SHARED_MEMORY,
                template_code="""
// SYCL local memory (equivalent to CUDA shared memory)
queue.submit([&](sycl::handler& cgh) {
    auto input_acc = input_buf.get_access<sycl::access::mode::read>(cgh);
    auto output_acc = output_buf.get_access<sycl::access::mode::write>(cgh);
    
    // Allocate local memory
    auto local_acc = sycl::local_accessor<float, 1>(sycl::range<1>(LOCAL_SIZE), cgh);
    
    cgh.parallel_for(sycl::nd_range<1>(global_size, local_size),
        [=](sycl::nd_item<1> item) {
        size_t lid = item.get_local_id(0);
        size_t gid = item.get_global_id(0);
        
        // Load to local memory
        local_acc[lid] = input_acc[gid];
        
        // Synchronize work-group
        item.barrier(sycl::access::fence_space::local_space);
        
        // Use local memory data
        output_acc[gid] = local_acc[lid] * 2.0f;
    });
});
""",
                includes=["<sycl/sycl.hpp>"],
                notes="Use sycl::local_accessor for work-group local memory"
            ),
            
            OptimizationPattern.WARP_SHUFFLE: SyclCodeTemplate(
                pattern_type=OptimizationPattern.WARP_SHUFFLE,
                template_code="""
// SYCL sub-group shuffle (equivalent to CUDA warp shuffle)
queue.submit([&](sycl::handler& cgh) {
    auto input_acc = input_buf.get_access<sycl::access::mode::read>(cgh);
    auto output_acc = output_buf.get_access<sycl::access::mode::write>(cgh);
    
    cgh.parallel_for(sycl::nd_range<1>(global_size, local_size),
        [=](sycl::nd_item<1> item) {
        auto sg = item.get_sub_group();
        size_t gid = item.get_global_id(0);
        
        float value = input_acc[gid];
        
        // Shuffle within sub-group (warp)
        value = sycl::shift_group_left(sg, value, 1);
        // Or: value = sycl::group_broadcast(sg, value, 0);
        
        output_acc[gid] = value;
    });
});
""",
                includes=["<sycl/sycl.hpp>"],
                notes="Sub-groups in SYCL are equivalent to CUDA warps"
            ),
            
            OptimizationPattern.TENSOR_CORE: SyclCodeTemplate(
                pattern_type=OptimizationPattern.TENSOR_CORE,
                template_code="""
// SYCL joint_matrix for tensor core operations
#include <sycl/ext/oneapi/experimental/matrix/matrix.hpp>
namespace matrix = sycl::ext::oneapi::experimental::matrix;

queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(sycl::nd_range<2>(global_range, local_range),
        [=](sycl::nd_item<2> item) {
        auto sg = item.get_sub_group();
        
        // Define matrix types (similar to CUDA wmma)
        matrix::joint_matrix<matrix::use::a, float, /* M */ 16, /* K */ 16> sub_a;
        matrix::joint_matrix<matrix::use::b, float, /* K */ 16, /* N */ 16> sub_b;
        matrix::joint_matrix<matrix::use::accumulator, float, 16, 16> sub_c;
        
        // Load matrices
        matrix::joint_matrix_load(sg, sub_a, a_ptr, stride_a);
        matrix::joint_matrix_load(sg, sub_b, b_ptr, stride_b);
        matrix::joint_matrix_fill(sg, sub_c, 0.0f);
        
        // Perform matrix multiply-accumulate
        sub_c = matrix::joint_matrix_mad(sg, sub_a, sub_b, sub_c);
        
        // Store result
        matrix::joint_matrix_store(sg, sub_c, c_ptr, stride_c);
    });
});
""",
                includes=[
                    "<sycl/sycl.hpp>",
                    "<sycl/ext/oneapi/experimental/matrix/matrix.hpp>"
                ],
                notes="joint_matrix extension provides tensor core access"
            ),
        }
        
    def _load_translation_rules(self) -> Dict:
        """
        Load CUDA to SYCL translation rules.
        
        Returns:
            Dictionary of translation rules
        """
        return {
            "cuda_concepts": {
                "thread": "work-item",
                "block": "work-group",
                "grid": "nd-range",
                "warp": "sub-group",
                "__shared__": "local_accessor",
                "__syncthreads()": "item.barrier()",
                "threadIdx": "item.get_local_id()",
                "blockIdx": "item.get_group_id()",
                "blockDim": "item.get_local_range()",
                "gridDim": "item.get_group_range()",
            },
            "memory_types": {
                "global": "global memory (default)",
                "shared": "local memory (local_accessor)",
                "constant": "constant memory (read-only accessor)",
                "texture": "image/sampler",
            }
        }
        
    async def generate_recommendations(
        self,
        analysis_report: OptimizationReport,
        target_repo: str = "my-org/sycl-kernels"
    ) -> RecommendationReport:
        """
        Generate SYCL recommendations from CUDA analysis.
        
        Args:
            analysis_report: Analysis of CUDA changes
            target_repo: Target SYCL repository
            
        Returns:
            Complete recommendation report
        """
        logger.info(f"Generating SYCL recommendations for {analysis_report.change_id}")
        
        recommendations = []
        
        for pattern in analysis_report.patterns:
            rec = self._create_recommendation(
                pattern,
                analysis_report,
                len(recommendations) + 1
            )
            if rec:
                recommendations.append(rec)
                
        # Sort by priority
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        recommendations.sort(key=lambda r: priority_order.get(r.priority, 4))
        
        # Determine overall priority
        overall_priority = "low"
        if recommendations:
            overall_priority = recommendations[0].priority
            
        # Generate summary
        summary = self._generate_report_summary(
            recommendations,
            analysis_report
        )
        
        return RecommendationReport(
            cuda_change_id=analysis_report.change_id,
            recommendations=recommendations,
            overall_priority=overall_priority,
            summary=summary,
            links={
                "cuda_pr": f"https://github.com/{analysis_report.change_id}",
                "target_repo": f"https://github.com/{target_repo}",
            }
        )
        
    def _create_recommendation(
        self,
        pattern,
        analysis_report: OptimizationReport,
        rec_number: int
    ) -> Optional[SyclRecommendation]:
        """
        Create a single SYCL recommendation from a pattern.
        
        Args:
            pattern: Code pattern from analysis
            analysis_report: Full analysis report
            rec_number: Recommendation number
            
        Returns:
            SYCL recommendation or None
        """
        template = self.templates.get(pattern.pattern_type)
        if not template:
            logger.warning(f"No template for {pattern.pattern_type}")
            return None
            
        # Determine priority based on pattern and confidence
        priority = self._determine_priority(pattern, analysis_report)
        
        # Generate title
        title = self._generate_recommendation_title(pattern.pattern_type)
        
        # Generate description
        description = self._generate_description(pattern, analysis_report)
        
        # Determine affected files (would be based on target repo structure)
        affected_files = ["src/kernels/optimized_kernel.cpp"]
        
        # Estimate implementation effort
        effort = self._estimate_effort(pattern.pattern_type, analysis_report.complexity)
        
        # Assess risk
        risk = self._assess_risk(pattern.pattern_type, pattern.confidence)
        
        # Generate test strategy
        test_strategy = self._generate_test_strategy(pattern.pattern_type)
        
        return SyclRecommendation(
            id=f"{analysis_report.change_id}-rec-{rec_number}",
            priority=priority,
            pattern=pattern.pattern_type,
            title=title,
            description=description,
            sycl_code=template.template_code,
            affected_files=affected_files,
            expected_impact=analysis_report.performance_impact.expected_speedup if analysis_report.performance_impact else "Unknown",
            implementation_effort=effort,
            risk_level=risk,
            test_strategy=test_strategy,
            additional_notes=template.notes
        )
        
    def _determine_priority(self, pattern, analysis_report: OptimizationReport) -> str:
        """Determine recommendation priority."""
        if pattern.confidence >= 0.9:
            if analysis_report.sycl_applicability >= 0.8:
                return "high"
        elif pattern.confidence >= 0.75:
            if analysis_report.sycl_applicability >= 0.7:
                return "medium"
        return "low"
        
    def _generate_recommendation_title(self, pattern_type: OptimizationPattern) -> str:
        """Generate a clear title for the recommendation."""
        titles = {
            OptimizationPattern.MEMORY_COALESCING: "Implement Memory Coalescing Optimization",
            OptimizationPattern.SHARED_MEMORY: "Add Local Memory (Shared Memory) Usage",
            OptimizationPattern.WARP_SHUFFLE: "Use Sub-group Shuffle Operations",
            OptimizationPattern.TENSOR_CORE: "Leverage Joint Matrix for Tensor Operations",
            OptimizationPattern.LOOP_UNROLLING: "Apply Loop Unrolling",
            OptimizationPattern.ASYNC_COPY: "Implement Asynchronous Memory Copies",
        }
        return titles.get(pattern_type, f"Optimize {pattern_type.value}")
        
    def _generate_description(self, pattern, analysis_report: OptimizationReport) -> str:
        """Generate detailed description of the recommendation."""
        desc = f"This optimization adapts the CUDA {pattern.pattern_type.value} pattern to SYCL. "
        desc += f"{pattern.description} "
        
        if analysis_report.performance_impact:
            desc += f"Expected performance improvement: {analysis_report.performance_impact.expected_speedup}. "
            
        return desc
        
    def _estimate_effort(self, pattern_type: OptimizationPattern, complexity: str) -> str:
        """Estimate implementation effort."""
        base_effort = {
            OptimizationPattern.MEMORY_COALESCING: 1,
            OptimizationPattern.SHARED_MEMORY: 2,
            OptimizationPattern.WARP_SHUFFLE: 3,
            OptimizationPattern.TENSOR_CORE: 5,
            OptimizationPattern.LOOP_UNROLLING: 1,
        }
        
        days = base_effort.get(pattern_type, 2)
        
        if complexity == "high":
            days *= 1.5
        elif complexity == "low":
            days *= 0.7
            
        if days <= 1:
            return "1 day"
        elif days <= 3:
            return f"{int(days)} days"
        else:
            return f"{int(days)} days - 1 week"
            
    def _assess_risk(self, pattern_type: OptimizationPattern, confidence: float) -> str:
        """Assess implementation risk."""
        if confidence >= 0.85:
            return "low"
        elif confidence >= 0.70:
            return "medium"
        else:
            return "high"
            
    def _generate_test_strategy(self, pattern_type: OptimizationPattern) -> str:
        """Generate testing strategy for the optimization."""
        strategies = {
            OptimizationPattern.MEMORY_COALESCING: 
                "1. Correctness test with various input sizes\n"
                "2. Performance benchmark comparing before/after\n"
                "3. Profiler analysis to verify coalesced access",
            OptimizationPattern.SHARED_MEMORY:
                "1. Correctness test with different work-group sizes\n"
                "2. Verify synchronization behavior\n"
                "3. Performance benchmark with memory-bound workload",
            OptimizationPattern.WARP_SHUFFLE:
                "1. Unit test for sub-group operations\n"
                "2. Verify correct data exchange between work-items\n"
                "3. Performance comparison with barrier-based approach",
        }
        return strategies.get(
            pattern_type,
            "1. Functional correctness tests\n2. Performance benchmarks\n3. Profiler validation"
        )
        
    def _generate_report_summary(
        self,
        recommendations: List[SyclRecommendation],
        analysis_report: OptimizationReport
    ) -> str:
        """Generate overall summary of recommendations."""
        if not recommendations:
            return "No actionable SYCL recommendations generated."
            
        summary = f"Generated {len(recommendations)} SYCL optimization recommendation(s) "
        summary += f"based on CUDA change analysis. "
        
        high_priority = sum(1 for r in recommendations if r.priority in ["critical", "high"])
        if high_priority > 0:
            summary += f"{high_priority} high-priority recommendation(s) identified. "
            
        summary += f"Overall SYCL applicability score: {analysis_report.sycl_applicability:.2f}. "
        
        return summary


# Example usage
if __name__ == "__main__":
    import asyncio
    from .analyzer import OptimizationAnalyzer, CodePattern, PerformanceImpact, OptimizationReport
    
    async def main():
        # Create mock analysis report
        patterns = [
            CodePattern(
                pattern_type=OptimizationPattern.MEMORY_COALESCING,
                confidence=0.90,
                location="kernel.cu:45",
                code_snippet="float val = input[threadIdx.x + offset * blockDim.x];",
                description="Memory access reorganized for coalescing"
            ),
            CodePattern(
                pattern_type=OptimizationPattern.SHARED_MEMORY,
                confidence=0.85,
                location="kernel.cu:52",
                code_snippet="__shared__ float smem[256];",
                description="Added shared memory for data reuse"
            )
        ]
        
        analysis_report = OptimizationReport(
            change_id="pytorch/pytorch#12345",
            patterns=patterns,
            performance_impact=PerformanceImpact(
                expected_speedup="15-20%",
                confidence=0.88,
                metric="kernel throughput",
                reasoning="Memory optimizations detected"
            ),
            complexity="medium",
            sycl_applicability=0.90,
            summary="Detected memory optimization patterns",
            recommendations=[]
        )
        
        # Generate SYCL recommendations
        recommender = SyclRecommender()
        report = await recommender.generate_recommendations(analysis_report)
        
        print(f"Summary: {report.summary}")
        print(f"Overall Priority: {report.overall_priority}")
        print(f"\nRecommendations ({len(report.recommendations)}):")
        for i, rec in enumerate(report.recommendations, 1):
            print(f"\n{i}. {rec.title}")
            print(f"   Priority: {rec.priority}")
            print(f"   Expected Impact: {rec.expected_impact}")
            print(f"   Effort: {rec.implementation_effort}")
            print(f"   Risk: {rec.risk_level}")
            
    asyncio.run(main())
