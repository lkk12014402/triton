"""
PR Generator Module

Automatically generates pull requests with SYCL optimizations.
"""

import logging
from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime

from .recommender import RecommendationReport, SyclRecommendation

logger = logging.getLogger(__name__)


@dataclass
class PRConfig:
    """Configuration for PR generation"""
    target_owner: str
    target_repo: str
    base_branch: str = "main"
    draft_mode: bool = True
    assign_reviewers: List[str] = None
    labels: List[str] = None
    auto_create_branch: bool = True


@dataclass
class GeneratedPR:
    """Information about a generated PR"""
    pr_number: Optional[int]
    branch_name: str
    title: str
    body: str
    files_changed: List[str]
    status: str  # "created", "draft", "failed"
    url: Optional[str]


class PRGenerator:
    """
    Generates pull requests with SYCL optimizations.
    
    This class handles:
    - Creating feature branches
    - Applying code changes
    - Generating tests
    - Creating PRs with detailed descriptions
    - Linking to source CUDA PRs
    
    Example:
        >>> generator = PRGenerator(github_token="ghp_xxxxx")
        >>> pr = await generator.create_pr(
        ...     recommendation_report=report,
        ...     config=pr_config
        ... )
        >>> print(f"Created PR: {pr.url}")
    """
    
    def __init__(self, github_token: str):
        """
        Initialize PR generator.
        
        Args:
            github_token: GitHub API token for PR creation
        """
        self.github_token = github_token
        
    async def create_pr(
        self,
        recommendation_report: RecommendationReport,
        config: PRConfig,
        selected_recommendations: Optional[List[str]] = None
    ) -> GeneratedPR:
        """
        Create a pull request with SYCL optimizations.
        
        Args:
            recommendation_report: Recommendations to implement
            config: PR configuration
            selected_recommendations: Optional list of recommendation IDs to include
                                    (if None, includes all high-priority recommendations)
            
        Returns:
            Information about the created PR
        """
        logger.info(f"Creating PR for {recommendation_report.cuda_change_id}")
        
        # Filter recommendations to include
        recommendations = self._filter_recommendations(
            recommendation_report.recommendations,
            selected_recommendations
        )
        
        if not recommendations:
            logger.warning("No recommendations selected for PR")
            return GeneratedPR(
                pr_number=None,
                branch_name="",
                title="",
                body="",
                files_changed=[],
                status="failed",
                url=None
            )
            
        # Generate branch name
        branch_name = self._generate_branch_name(recommendation_report.cuda_change_id)
        
        # Generate PR title
        title = self._generate_pr_title(recommendations, recommendation_report)
        
        # Generate PR body
        body = self._generate_pr_body(
            recommendations,
            recommendation_report,
            config
        )
        
        # Collect all affected files
        files_changed = self._collect_affected_files(recommendations)
        
        # In a real implementation:
        # 1. Create branch
        # 2. Generate and commit code changes
        # 3. Push to remote
        # 4. Create PR via GitHub API
        
        logger.info(f"Generated PR: {title}")
        logger.info(f"Branch: {branch_name}")
        logger.info(f"Files changed: {len(files_changed)}")
        
        return GeneratedPR(
            pr_number=None,  # Would be set after API call
            branch_name=branch_name,
            title=title,
            body=body,
            files_changed=files_changed,
            status="draft" if config.draft_mode else "created",
            url=None  # Would be set after API call
        )
        
    def _filter_recommendations(
        self,
        all_recommendations: List[SyclRecommendation],
        selected_ids: Optional[List[str]]
    ) -> List[SyclRecommendation]:
        """
        Filter recommendations to include in PR.
        
        Args:
            all_recommendations: All available recommendations
            selected_ids: Specific IDs to include (or None for auto-selection)
            
        Returns:
            Filtered list of recommendations
        """
        if selected_ids:
            return [r for r in all_recommendations if r.id in selected_ids]
        else:
            # Auto-select high and critical priority recommendations
            return [
                r for r in all_recommendations
                if r.priority in ["critical", "high"]
            ]
            
    def _generate_branch_name(self, change_id: str) -> str:
        """
        Generate a branch name for the PR.
        
        Args:
            change_id: CUDA change identifier
            
        Returns:
            Branch name
        """
        # Clean up change ID for branch name
        safe_id = change_id.replace("/", "-").replace("#", "-")
        timestamp = datetime.now().strftime("%Y%m%d")
        return f"sycl-opt/{safe_id}-{timestamp}"
        
    def _generate_pr_title(
        self,
        recommendations: List[SyclRecommendation],
        report: RecommendationReport
    ) -> str:
        """
        Generate PR title.
        
        Args:
            recommendations: Recommendations to include
            report: Full recommendation report
            
        Returns:
            PR title
        """
        if len(recommendations) == 1:
            return f"[SYCL] {recommendations[0].title}"
        else:
            return f"[SYCL] Apply {len(recommendations)} optimization(s) from CUDA PR"
            
    def _generate_pr_body(
        self,
        recommendations: List[SyclRecommendation],
        report: RecommendationReport,
        config: PRConfig
    ) -> str:
        """
        Generate detailed PR description.
        
        Args:
            recommendations: Recommendations to include
            report: Full recommendation report
            config: PR configuration
            
        Returns:
            PR body in Markdown format
        """
        body = "## CUDA Optimization Adaptation\n\n"
        
        # Source information
        body += "### Source\n"
        body += f"- **CUDA Change**: [{report.cuda_change_id}]({report.links.get('cuda_pr', '#')})\n"
        body += f"- **Detection Date**: {datetime.now().strftime('%Y-%m-%d')}\n"
        body += f"- **Number of Optimizations**: {len(recommendations)}\n\n"
        
        # Summary
        body += "### Summary\n"
        body += f"{report.summary}\n\n"
        
        # Optimizations
        body += "### Optimizations Applied\n\n"
        for i, rec in enumerate(recommendations, 1):
            body += f"#### {i}. {rec.title}\n\n"
            body += f"**Priority**: {rec.priority.upper()}  \n"
            body += f"**Expected Impact**: {rec.expected_impact}  \n"
            body += f"**Implementation Effort**: {rec.implementation_effort}  \n"
            body += f"**Risk Level**: {rec.risk_level}  \n\n"
            body += f"**Description**: {rec.description}\n\n"
            
            if rec.affected_files:
                body += "**Files Changed**:\n"
                for file in rec.affected_files:
                    body += f"- `{file}`\n"
                body += "\n"
                
        # SYCL-specific adaptations
        body += "### SYCL-Specific Adaptations\n\n"
        body += self._generate_adaptations_section(recommendations)
        body += "\n"
        
        # Testing
        body += "### Testing Strategy\n\n"
        body += self._generate_testing_section(recommendations)
        body += "\n"
        
        # Checklist
        body += "### Verification Checklist\n\n"
        body += "- [ ] Code compiles successfully\n"
        body += "- [ ] Unit tests pass\n"
        body += "- [ ] Performance benchmarks show improvement\n"
        body += "- [ ] Correctness validation complete\n"
        body += "- [ ] Code review completed\n"
        body += "- [ ] Documentation updated\n\n"
        
        # Additional notes
        body += "### Additional Notes\n\n"
        body += "This PR was generated automatically by the CUDA-to-SYCL optimization agent. "
        body += "Please review the changes carefully and run the full test suite before merging.\n\n"
        
        # Related links
        if report.links:
            body += "### Related Links\n\n"
            for key, url in report.links.items():
                body += f"- [{key}]({url})\n"
            body += "\n"
            
        return body
        
    def _generate_adaptations_section(
        self,
        recommendations: List[SyclRecommendation]
    ) -> str:
        """Generate SYCL adaptations section."""
        adaptations = []
        
        for rec in recommendations:
            if "CUDA" in rec.description or "SYCL" in rec.description:
                adaptations.append(f"- {rec.pattern.value}: {rec.additional_notes}")
                
        if adaptations:
            return "\n".join(adaptations)
        else:
            return "- Standard CUDA-to-SYCL pattern translations applied"
            
    def _generate_testing_section(
        self,
        recommendations: List[SyclRecommendation]
    ) -> str:
        """Generate testing section."""
        testing = "**Test Plan**:\n\n"
        
        for i, rec in enumerate(recommendations, 1):
            testing += f"{i}. **{rec.title}**:\n"
            # Indent test strategy
            for line in rec.test_strategy.split('\n'):
                if line.strip():
                    testing += f"   {line}\n"
            testing += "\n"
            
        return testing
        
    def _collect_affected_files(
        self,
        recommendations: List[SyclRecommendation]
    ) -> List[str]:
        """Collect all affected files from recommendations."""
        files = set()
        for rec in recommendations:
            files.update(rec.affected_files)
        return sorted(list(files))
        
    async def generate_code_changes(
        self,
        recommendation: SyclRecommendation,
        target_file: str
    ) -> str:
        """
        Generate actual code changes for a recommendation.
        
        Args:
            recommendation: Recommendation to implement
            target_file: File to modify
            
        Returns:
            Modified file content
        """
        # In a real implementation:
        # 1. Read target file
        # 2. Parse and understand existing code
        # 3. Insert/modify code based on recommendation
        # 4. Return modified content
        
        logger.info(f"Generating code for {target_file}")
        return recommendation.sycl_code
        
    async def generate_test_code(
        self,
        recommendation: SyclRecommendation
    ) -> str:
        """
        Generate test code for a recommendation.
        
        Args:
            recommendation: Recommendation to test
            
        Returns:
            Test code
        """
        test_template = f"""
// Test for {recommendation.title}
#include <sycl/sycl.hpp>
#include <cassert>
#include <iostream>

void test_{recommendation.pattern.value}() {{
    sycl::queue queue;
    
    // Setup test data
    constexpr size_t N = 1024;
    std::vector<float> input(N);
    std::vector<float> output(N);
    
    // Initialize input
    for (size_t i = 0; i < N; ++i) {{
        input[i] = static_cast<float>(i);
    }}
    
    // Create buffers
    sycl::buffer<float> input_buf(input.data(), sycl::range<1>(N));
    sycl::buffer<float> output_buf(output.data(), sycl::range<1>(N));
    
    // Run kernel (implementation goes here)
    // ...
    
    // Verify results
    {{
        auto output_acc = output_buf.get_host_access();
        for (size_t i = 0; i < N; ++i) {{
            // Add verification logic
            assert(output_acc[i] >= 0.0f);
        }}
    }}
    
    std::cout << "Test {recommendation.pattern.value} passed!" << std::endl;
}}

int main() {{
    try {{
        test_{recommendation.pattern.value}();
        return 0;
    }} catch (const std::exception& e) {{
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }}
}}
"""
        return test_template
        
    async def create_benchmark(
        self,
        recommendation: SyclRecommendation
    ) -> str:
        """
        Generate benchmark code for performance measurement.
        
        Args:
            recommendation: Recommendation to benchmark
            
        Returns:
            Benchmark code
        """
        benchmark_template = f"""
// Benchmark for {recommendation.title}
#include <sycl/sycl.hpp>
#include <chrono>
#include <iostream>

void benchmark_{recommendation.pattern.value}() {{
    sycl::queue queue;
    
    constexpr size_t N = 1 << 20;  // 1M elements
    constexpr int iterations = 100;
    
    std::vector<float> input(N, 1.0f);
    std::vector<float> output(N, 0.0f);
    
    sycl::buffer<float> input_buf(input.data(), sycl::range<1>(N));
    sycl::buffer<float> output_buf(output.data(), sycl::range<1>(N));
    
    // Warmup
    for (int i = 0; i < 10; ++i) {{
        // Run kernel
    }}
    
    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; ++i) {{
        // Run kernel
    }}
    
    queue.wait();
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double avg_time_ms = duration.count() / (1000.0 * iterations);
    double throughput_gb_s = (N * sizeof(float) * 2) / (avg_time_ms * 1e6);  // Read + Write
    
    std::cout << "Benchmark {recommendation.pattern.value}:" << std::endl;
    std::cout << "  Average time: " << avg_time_ms << " ms" << std::endl;
    std::cout << "  Throughput: " << throughput_gb_s << " GB/s" << std::endl;
}}

int main() {{
    benchmark_{recommendation.pattern.value}();
    return 0;
}}
"""
        return benchmark_template


# Example usage
if __name__ == "__main__":
    import asyncio
    from .recommender import SyclRecommender, RecommendationReport, SyclRecommendation
    from .analyzer import OptimizationPattern
    
    async def main():
        # Create mock recommendation
        rec = SyclRecommendation(
            id="test-rec-1",
            priority="high",
            pattern=OptimizationPattern.MEMORY_COALESCING,
            title="Implement Memory Coalescing",
            description="Optimize memory access patterns",
            sycl_code="// SYCL code here",
            affected_files=["src/kernel.cpp"],
            expected_impact="15-20%",
            implementation_effort="2 days",
            risk_level="low",
            test_strategy="Unit tests + benchmarks",
            additional_notes="Direct translation"
        )
        
        report = RecommendationReport(
            cuda_change_id="pytorch/pytorch#12345",
            recommendations=[rec],
            overall_priority="high",
            summary="One high-priority optimization",
            links={"cuda_pr": "https://github.com/pytorch/pytorch/pull/12345"}
        )
        
        config = PRConfig(
            target_owner="my-org",
            target_repo="sycl-kernels",
            base_branch="main",
            draft_mode=True,
            labels=["optimization", "auto-generated"]
        )
        
        generator = PRGenerator(github_token="dummy-token")
        pr = await generator.create_pr(report, config)
        
        print(f"PR Title: {pr.title}")
        print(f"Branch: {pr.branch_name}")
        print(f"\n{pr.body}")
        
    asyncio.run(main())
