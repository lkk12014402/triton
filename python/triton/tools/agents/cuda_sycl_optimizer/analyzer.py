"""
Optimization Analyzer Module

Analyzes CUDA kernel changes to identify optimization patterns and techniques.
"""

import logging
from dataclasses import dataclass
from typing import List, Dict, Optional, Set
from enum import Enum

logger = logging.getLogger(__name__)


class OptimizationPattern(Enum):
    """Types of optimization patterns"""
    MEMORY_COALESCING = "memory_coalescing"
    SHARED_MEMORY = "shared_memory"
    REGISTER_OPTIMIZATION = "register_optimization"
    LOOP_UNROLLING = "loop_unrolling"
    WARP_SHUFFLE = "warp_shuffle"
    TENSOR_CORE = "tensor_core"
    ASYNC_COPY = "async_copy"
    BARRIER_REDUCTION = "barrier_reduction"
    BANK_CONFLICT_AVOIDANCE = "bank_conflict_avoidance"
    OCCUPANCY_TUNING = "occupancy_tuning"


@dataclass
class CodePattern:
    """Represents a code pattern found in analysis"""
    pattern_type: OptimizationPattern
    confidence: float
    location: str
    code_snippet: str
    description: str


@dataclass
class PerformanceImpact:
    """Estimated performance impact of an optimization"""
    expected_speedup: str  # e.g., "15-20%"
    confidence: float
    metric: str  # e.g., "throughput", "latency"
    reasoning: str


@dataclass
class OptimizationReport:
    """Complete analysis report for a kernel change"""
    change_id: str
    patterns: List[CodePattern]
    performance_impact: Optional[PerformanceImpact]
    complexity: str  # low, medium, high
    sycl_applicability: float  # 0-1 score
    summary: str
    recommendations: List[str]


class OptimizationAnalyzer:
    """
    Analyzes CUDA kernel changes to identify optimization patterns.
    
    This class performs deep analysis of CUDA code changes to:
    - Identify optimization patterns
    - Estimate performance impact
    - Assess applicability to SYCL
    - Generate analysis reports
    
    Example:
        >>> analyzer = OptimizationAnalyzer()
        >>> report = await analyzer.analyze_change(change_id="cuda-pr-12345")
        >>> print(f"Found {len(report.patterns)} optimization patterns")
    """
    
    def __init__(self):
        """Initialize the analyzer with pattern database."""
        self.pattern_signatures = self._load_pattern_signatures()
        
    def _load_pattern_signatures(self) -> Dict[OptimizationPattern, Dict]:
        """
        Load known optimization pattern signatures.
        
        Returns:
            Dictionary mapping pattern types to their signatures
        """
        # In production, load from database or configuration file
        return {
            OptimizationPattern.MEMORY_COALESCING: {
                'keywords': ['coalesce', 'stride', 'alignment'],
                'ast_patterns': [],
                'heuristics': ['sequential_access', 'aligned_offset']
            },
            OptimizationPattern.SHARED_MEMORY: {
                'keywords': ['__shared__', 'smem', 'shared memory'],
                'ast_patterns': [],
                'heuristics': ['shared_declaration', 'sync_threads']
            },
            OptimizationPattern.WARP_SHUFFLE: {
                'keywords': ['__shfl', 'shuffle', 'warp'],
                'ast_patterns': [],
                'heuristics': ['shuffle_instruction']
            },
            OptimizationPattern.TENSOR_CORE: {
                'keywords': ['wmma', 'mma', 'tensor core'],
                'ast_patterns': [],
                'heuristics': ['wmma_namespace']
            },
        }
        
    async def analyze_change(
        self,
        change_id: str,
        old_code: Optional[str] = None,
        new_code: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> OptimizationReport:
        """
        Analyze a kernel change to identify optimizations.
        
        Args:
            change_id: Unique identifier for the change
            old_code: Previous version of the code
            new_code: New version of the code
            metadata: Additional metadata (PR description, comments, etc.)
            
        Returns:
            Complete optimization analysis report
        """
        logger.info(f"Analyzing change {change_id}")
        
        # Extract patterns
        patterns = []
        if old_code and new_code:
            patterns = await self._extract_patterns(old_code, new_code)
        
        # Estimate performance impact
        perf_impact = None
        if patterns:
            perf_impact = self._estimate_performance_impact(patterns, metadata)
        
        # Assess SYCL applicability
        sycl_score = self._assess_sycl_applicability(patterns)
        
        # Determine implementation complexity
        complexity = self._estimate_complexity(patterns)
        
        # Generate summary
        summary = self._generate_summary(patterns, perf_impact, metadata)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(patterns, sycl_score)
        
        return OptimizationReport(
            change_id=change_id,
            patterns=patterns,
            performance_impact=perf_impact,
            complexity=complexity,
            sycl_applicability=sycl_score,
            summary=summary,
            recommendations=recommendations
        )
        
    async def _extract_patterns(
        self,
        old_code: str,
        new_code: str
    ) -> List[CodePattern]:
        """
        Extract optimization patterns from code diff.
        
        Args:
            old_code: Previous version
            new_code: New version
            
        Returns:
            List of identified patterns
        """
        patterns = []
        
        # Analyze memory patterns
        patterns.extend(self._analyze_memory_patterns(old_code, new_code))
        
        # Analyze compute patterns
        patterns.extend(self._analyze_compute_patterns(old_code, new_code))
        
        # Analyze synchronization patterns
        patterns.extend(self._analyze_sync_patterns(old_code, new_code))
        
        # Analyze architecture-specific patterns
        patterns.extend(self._analyze_arch_patterns(old_code, new_code))
        
        return patterns
        
    def _analyze_memory_patterns(
        self,
        old_code: str,
        new_code: str
    ) -> List[CodePattern]:
        """Analyze memory access optimization patterns."""
        patterns = []
        
        # Check for memory coalescing
        if self._detect_coalescing_optimization(old_code, new_code):
            patterns.append(CodePattern(
                pattern_type=OptimizationPattern.MEMORY_COALESCING,
                confidence=0.85,
                location="kernel",
                code_snippet=self._extract_relevant_snippet(new_code, "coalescing"),
                description="Memory access pattern reorganized for coalescing"
            ))
            
        # Check for shared memory usage
        if "__shared__" in new_code and "__shared__" not in old_code:
            patterns.append(CodePattern(
                pattern_type=OptimizationPattern.SHARED_MEMORY,
                confidence=0.90,
                location="kernel",
                code_snippet=self._extract_relevant_snippet(new_code, "__shared__"),
                description="Added shared memory usage for data reuse"
            ))
            
        # Check for bank conflict avoidance
        if "bank conflict" in new_code.lower() or self._detect_padding(new_code):
            patterns.append(CodePattern(
                pattern_type=OptimizationPattern.BANK_CONFLICT_AVOIDANCE,
                confidence=0.75,
                location="kernel",
                code_snippet=self._extract_relevant_snippet(new_code, "padding"),
                description="Added padding to avoid shared memory bank conflicts"
            ))
            
        return patterns
        
    def _analyze_compute_patterns(
        self,
        old_code: str,
        new_code: str
    ) -> List[CodePattern]:
        """Analyze computation optimization patterns."""
        patterns = []
        
        # Check for loop unrolling
        if "#pragma unroll" in new_code and "#pragma unroll" not in old_code:
            patterns.append(CodePattern(
                pattern_type=OptimizationPattern.LOOP_UNROLLING,
                confidence=0.88,
                location="kernel",
                code_snippet=self._extract_relevant_snippet(new_code, "#pragma unroll"),
                description="Added loop unrolling for instruction-level parallelism"
            ))
            
        # Check for register optimization
        if "register" in new_code and self._detect_register_blocking(new_code):
            patterns.append(CodePattern(
                pattern_type=OptimizationPattern.REGISTER_OPTIMIZATION,
                confidence=0.80,
                location="kernel",
                code_snippet=self._extract_relevant_snippet(new_code, "register"),
                description="Optimized register usage through register blocking"
            ))
            
        return patterns
        
    def _analyze_sync_patterns(
        self,
        old_code: str,
        new_code: str
    ) -> List[CodePattern]:
        """Analyze synchronization optimization patterns."""
        patterns = []
        
        # Count synchronization points
        old_syncs = old_code.count("__syncthreads()")
        new_syncs = new_code.count("__syncthreads()")
        
        if new_syncs < old_syncs:
            patterns.append(CodePattern(
                pattern_type=OptimizationPattern.BARRIER_REDUCTION,
                confidence=0.85,
                location="kernel",
                code_snippet=f"Reduced barriers from {old_syncs} to {new_syncs}",
                description="Reduced synchronization overhead by removing barriers"
            ))
            
        # Check for warp-level operations
        if "__shfl" in new_code or "__ballot" in new_code:
            patterns.append(CodePattern(
                pattern_type=OptimizationPattern.WARP_SHUFFLE,
                confidence=0.90,
                location="kernel",
                code_snippet=self._extract_relevant_snippet(new_code, "__shfl"),
                description="Using warp shuffle operations for intra-warp communication"
            ))
            
        return patterns
        
    def _analyze_arch_patterns(
        self,
        old_code: str,
        new_code: str
    ) -> List[CodePattern]:
        """Analyze architecture-specific optimization patterns."""
        patterns = []
        
        # Check for tensor core usage
        if "wmma::" in new_code or "mma" in new_code.lower():
            patterns.append(CodePattern(
                pattern_type=OptimizationPattern.TENSOR_CORE,
                confidence=0.92,
                location="kernel",
                code_snippet=self._extract_relevant_snippet(new_code, "wmma"),
                description="Leveraging tensor cores for matrix operations"
            ))
            
        # Check for async copy operations
        if "cp.async" in new_code or "async_copy" in new_code:
            patterns.append(CodePattern(
                pattern_type=OptimizationPattern.ASYNC_COPY,
                confidence=0.88,
                location="kernel",
                code_snippet=self._extract_relevant_snippet(new_code, "async"),
                description="Using asynchronous memory copy operations"
            ))
            
        return patterns
        
    def _detect_coalescing_optimization(
        self,
        old_code: str,
        new_code: str
    ) -> bool:
        """Detect if memory coalescing was optimized."""
        # Simple heuristic: look for changes in indexing patterns
        old_has_stride = "threadIdx.x *" in old_code or "stride" in old_code
        new_has_sequential = "threadIdx.x +" in new_code or "+ threadIdx.x" in new_code
        
        return old_has_stride and new_has_sequential
        
    def _detect_padding(self, code: str) -> bool:
        """Detect padding added for bank conflict avoidance."""
        return "+ 1" in code or "PADDING" in code or "padding" in code
        
    def _detect_register_blocking(self, code: str) -> bool:
        """Detect register blocking pattern."""
        # Look for explicit register arrays or blocking patterns
        return "float reg[" in code or "double reg[" in code or "register float" in code
        
    def _extract_relevant_snippet(self, code: str, keyword: str, lines: int = 3) -> str:
        """
        Extract code snippet around a keyword.
        
        Args:
            code: Full code text
            keyword: Keyword to find
            lines: Number of lines to include around keyword
            
        Returns:
            Code snippet
        """
        code_lines = code.split('\n')
        for i, line in enumerate(code_lines):
            if keyword.lower() in line.lower():
                start = max(0, i - lines)
                end = min(len(code_lines), i + lines + 1)
                return '\n'.join(code_lines[start:end])
        return ""
        
    def _estimate_performance_impact(
        self,
        patterns: List[CodePattern],
        metadata: Optional[Dict]
    ) -> Optional[PerformanceImpact]:
        """
        Estimate the performance impact of identified optimizations.
        
        Args:
            patterns: List of identified patterns
            metadata: Additional context
            
        Returns:
            Performance impact estimation
        """
        if not patterns:
            return None
            
        # Simple scoring based on pattern types
        impact_scores = {
            OptimizationPattern.MEMORY_COALESCING: 0.20,  # 20% potential gain
            OptimizationPattern.SHARED_MEMORY: 0.25,
            OptimizationPattern.TENSOR_CORE: 0.40,
            OptimizationPattern.WARP_SHUFFLE: 0.15,
            OptimizationPattern.LOOP_UNROLLING: 0.10,
            OptimizationPattern.ASYNC_COPY: 0.18,
        }
        
        total_impact = sum(
            impact_scores.get(p.pattern_type, 0.05) * p.confidence
            for p in patterns
        )
        
        # Cap at reasonable maximum
        total_impact = min(total_impact, 0.50)
        
        # Convert to percentage range
        low = int(total_impact * 80)  # Lower bound
        high = int(total_impact * 120)  # Upper bound
        
        return PerformanceImpact(
            expected_speedup=f"{low}-{high}%",
            confidence=sum(p.confidence for p in patterns) / len(patterns),
            metric="overall kernel performance",
            reasoning=f"Based on {len(patterns)} optimization patterns"
        )
        
    def _assess_sycl_applicability(self, patterns: List[CodePattern]) -> float:
        """
        Assess how applicable the optimizations are to SYCL.
        
        Args:
            patterns: List of identified patterns
            
        Returns:
            Applicability score (0-1)
        """
        if not patterns:
            return 0.0
            
        # SYCL applicability by pattern type
        sycl_applicability = {
            OptimizationPattern.MEMORY_COALESCING: 0.95,  # Directly applicable
            OptimizationPattern.SHARED_MEMORY: 0.90,      # Maps to local accessor
            OptimizationPattern.WARP_SHUFFLE: 0.85,       # Maps to sub-group ops
            OptimizationPattern.LOOP_UNROLLING: 0.95,     # Directly applicable
            OptimizationPattern.TENSOR_CORE: 0.70,        # Joint matrix extension
            OptimizationPattern.ASYNC_COPY: 0.75,         # Async copy extension
            OptimizationPattern.BARRIER_REDUCTION: 0.90,  # Work-group barriers
        }
        
        scores = [
            sycl_applicability.get(p.pattern_type, 0.5) * p.confidence
            for p in patterns
        ]
        
        return sum(scores) / len(scores) if scores else 0.0
        
    def _estimate_complexity(self, patterns: List[CodePattern]) -> str:
        """
        Estimate implementation complexity for SYCL.
        
        Args:
            patterns: List of identified patterns
            
        Returns:
            Complexity level: "low", "medium", or "high"
        """
        if not patterns:
            return "low"
            
        # Complexity by pattern type
        complexity_scores = {
            OptimizationPattern.MEMORY_COALESCING: 2,
            OptimizationPattern.SHARED_MEMORY: 3,
            OptimizationPattern.LOOP_UNROLLING: 1,
            OptimizationPattern.WARP_SHUFFLE: 4,
            OptimizationPattern.TENSOR_CORE: 5,
            OptimizationPattern.ASYNC_COPY: 4,
        }
        
        avg_complexity = sum(
            complexity_scores.get(p.pattern_type, 2)
            for p in patterns
        ) / len(patterns)
        
        if avg_complexity <= 2:
            return "low"
        elif avg_complexity <= 3.5:
            return "medium"
        else:
            return "high"
            
    def _generate_summary(
        self,
        patterns: List[CodePattern],
        perf_impact: Optional[PerformanceImpact],
        metadata: Optional[Dict]
    ) -> str:
        """
        Generate a human-readable summary of the analysis.
        
        Args:
            patterns: Identified patterns
            perf_impact: Performance impact estimation
            metadata: Additional context
            
        Returns:
            Summary text
        """
        if not patterns:
            return "No significant optimization patterns detected."
            
        pattern_names = [p.pattern_type.value.replace('_', ' ').title() for p in patterns]
        pattern_list = ", ".join(pattern_names[:3])
        if len(pattern_names) > 3:
            pattern_list += f", and {len(pattern_names) - 3} more"
            
        summary = f"Detected {len(patterns)} optimization pattern(s): {pattern_list}. "
        
        if perf_impact:
            summary += f"Expected performance improvement: {perf_impact.expected_speedup}. "
            
        return summary
        
    def _generate_recommendations(
        self,
        patterns: List[CodePattern],
        sycl_score: float
    ) -> List[str]:
        """
        Generate actionable recommendations for SYCL implementation.
        
        Args:
            patterns: Identified patterns
            sycl_score: SYCL applicability score
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        if sycl_score >= 0.8:
            recommendations.append(
                "This optimization is highly applicable to SYCL and should be prioritized."
            )
        elif sycl_score >= 0.6:
            recommendations.append(
                "This optimization can be adapted to SYCL with moderate effort."
            )
        else:
            recommendations.append(
                "This optimization requires significant adaptation for SYCL."
            )
            
        for pattern in patterns:
            if pattern.pattern_type == OptimizationPattern.MEMORY_COALESCING:
                recommendations.append(
                    "Use appropriate work-item ordering to ensure coalesced access in SYCL."
                )
            elif pattern.pattern_type == OptimizationPattern.SHARED_MEMORY:
                recommendations.append(
                    "Map CUDA shared memory to SYCL local accessors."
                )
            elif pattern.pattern_type == OptimizationPattern.WARP_SHUFFLE:
                recommendations.append(
                    "Use SYCL sub-group shuffle operations as equivalent to warp shuffles."
                )
            elif pattern.pattern_type == OptimizationPattern.TENSOR_CORE:
                recommendations.append(
                    "Leverage SYCL joint_matrix extension for tensor core operations."
                )
                
        return recommendations


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def main():
        analyzer = OptimizationAnalyzer()
        
        old_code = """
        __global__ void kernel(float* input, float* output) {
            int idx = threadIdx.x * 128 + blockIdx.x;
            output[idx] = input[idx] * 2.0f;
        }
        """
        
        new_code = """
        __global__ void kernel(float* input, float* output) {
            __shared__ float smem[256];
            int idx = threadIdx.x + blockIdx.x * blockDim.x;
            smem[threadIdx.x] = input[idx];
            __syncthreads();
            output[idx] = smem[threadIdx.x] * 2.0f;
        }
        """
        
        report = await analyzer.analyze_change(
            change_id="example-1",
            old_code=old_code,
            new_code=new_code
        )
        
        print(f"Analysis Summary: {report.summary}")
        print(f"Patterns found: {len(report.patterns)}")
        print(f"SYCL Applicability: {report.sycl_applicability:.2f}")
        print(f"Complexity: {report.complexity}")
        
    asyncio.run(main())
