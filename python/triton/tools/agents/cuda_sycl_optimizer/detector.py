"""
Change Detector Module

Monitors GitHub repositories for CUDA kernel changes and updates.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Set
from enum import Enum

logger = logging.getLogger(__name__)


class ChangeType(Enum):
    """Types of kernel changes"""
    NEW_KERNEL = "new_kernel"
    OPTIMIZATION = "optimization"
    BUG_FIX = "bug_fix"
    API_CHANGE = "api_change"
    REFACTORING = "refactoring"


class Priority(Enum):
    """Priority levels for changes"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class KernelChange:
    """Represents a detected kernel change"""
    id: str
    repository: str
    pr_number: Optional[int]
    commit_sha: str
    title: str
    description: str
    change_type: ChangeType
    priority: Priority
    affected_files: List[str]
    diff_url: str
    author: str
    created_at: datetime
    labels: Set[str]
    performance_related: bool


@dataclass
class RepositoryConfig:
    """Configuration for a monitored repository"""
    owner: str
    repo: str
    branches: List[str]
    watch_patterns: List[str]
    include_labels: Set[str]
    exclude_labels: Set[str]


class ChangeDetector:
    """
    Detects and monitors CUDA kernel changes across GitHub repositories.
    
    This class provides functionality to:
    - Monitor GitHub repositories via API polling or webhooks
    - Filter changes based on file patterns and labels
    - Classify changes by type and priority
    - Queue changes for further analysis
    
    Example:
        >>> detector = ChangeDetector(api_token="ghp_xxxxx")
        >>> detector.add_repository(
        ...     owner="pytorch",
        ...     repo="pytorch",
        ...     branches=["main"],
        ...     watch_patterns=["*.cu", "*.cuh"]
        ... )
        >>> await detector.start_monitoring()
    """
    
    def __init__(self, api_token: str, poll_interval: int = 3600):
        """
        Initialize the change detector.
        
        Args:
            api_token: GitHub API token for authentication
            poll_interval: Seconds between repository polls (default: 3600)
        """
        self.api_token = api_token
        self.poll_interval = poll_interval
        self.repositories: List[RepositoryConfig] = []
        self._monitoring = False
        self._last_check: Dict[str, datetime] = {}
        
    def add_repository(
        self,
        owner: str,
        repo: str,
        branches: List[str] = None,
        watch_patterns: List[str] = None,
        include_labels: Set[str] = None,
        exclude_labels: Set[str] = None
    ):
        """
        Add a repository to monitor.
        
        Args:
            owner: Repository owner (user or organization)
            repo: Repository name
            branches: Branches to monitor (default: ["main"])
            watch_patterns: File patterns to watch (default: ["*.cu", "*.cuh"])
            include_labels: Labels to include (default: {"performance", "optimization"})
            exclude_labels: Labels to exclude (default: {"documentation"})
        """
        if branches is None:
            branches = ["main"]
        if watch_patterns is None:
            watch_patterns = ["*.cu", "*.cuh"]
        if include_labels is None:
            include_labels = {"performance", "optimization", "cuda"}
        if exclude_labels is None:
            exclude_labels = {"documentation", "ci", "wip"}
            
        config = RepositoryConfig(
            owner=owner,
            repo=repo,
            branches=branches,
            watch_patterns=watch_patterns,
            include_labels=include_labels,
            exclude_labels=exclude_labels
        )
        self.repositories.append(config)
        logger.info(f"Added repository {owner}/{repo} for monitoring")
        
    async def start_monitoring(self):
        """
        Start continuous monitoring of configured repositories.
        
        This method runs indefinitely, checking for changes at the specified
        poll interval.
        """
        self._monitoring = True
        logger.info("Starting change detection monitoring")
        
        while self._monitoring:
            try:
                await self._check_all_repositories()
                await asyncio.sleep(self.poll_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}", exc_info=True)
                await asyncio.sleep(60)  # Wait a minute before retrying
                
    def stop_monitoring(self):
        """Stop the monitoring loop."""
        self._monitoring = False
        logger.info("Stopped change detection monitoring")
        
    async def _check_all_repositories(self):
        """Check all configured repositories for changes."""
        for repo_config in self.repositories:
            try:
                changes = await self._check_repository(repo_config)
                for change in changes:
                    await self._process_change(change)
            except Exception as e:
                logger.error(
                    f"Error checking {repo_config.owner}/{repo_config.repo}: {e}",
                    exc_info=True
                )
                
    async def _check_repository(
        self,
        repo_config: RepositoryConfig
    ) -> List[KernelChange]:
        """
        Check a single repository for changes.
        
        Args:
            repo_config: Repository configuration
            
        Returns:
            List of detected kernel changes
        """
        repo_key = f"{repo_config.owner}/{repo_config.repo}"
        since = self._last_check.get(repo_key, datetime.now() - timedelta(days=7))
        
        changes = []
        
        # In a real implementation, this would use the GitHub API
        # For now, this is a placeholder
        logger.info(f"Checking {repo_key} for changes since {since}")
        
        # TODO: Implement GitHub API calls to fetch:
        # 1. Recent pull requests
        # 2. Recent commits
        # 3. Filter by labels and file patterns
        # 4. Create KernelChange objects
        
        self._last_check[repo_key] = datetime.now()
        return changes
        
    def _is_relevant_change(
        self,
        change: Dict,
        repo_config: RepositoryConfig
    ) -> bool:
        """
        Determine if a change is relevant based on filters.
        
        Args:
            change: Raw change data from GitHub API
            repo_config: Repository configuration with filters
            
        Returns:
            True if change should be processed
        """
        # Check labels
        change_labels = set(change.get('labels', []))
        
        # Exclude if has any exclude labels
        if change_labels & repo_config.exclude_labels:
            return False
            
        # Include if has any include labels
        if not (change_labels & repo_config.include_labels):
            return False
            
        # Check file patterns
        files = change.get('files', [])
        for file in files:
            filename = file.get('filename', '')
            for pattern in repo_config.watch_patterns:
                if self._matches_pattern(filename, pattern):
                    return True
                    
        return False
        
    def _matches_pattern(self, filename: str, pattern: str) -> bool:
        """
        Check if filename matches a pattern.
        
        Args:
            filename: File path/name
            pattern: Pattern to match (e.g., "*.cu")
            
        Returns:
            True if filename matches pattern
        """
        # Simple pattern matching - in production, use fnmatch or glob
        if pattern.startswith("**"):
            return filename.endswith(pattern[2:].lstrip('/'))
        elif pattern.startswith("*"):
            return filename.endswith(pattern[1:])
        else:
            return pattern in filename
            
    def _classify_change(self, change: Dict) -> ChangeType:
        """
        Classify the type of change based on metadata.
        
        Args:
            change: Change metadata
            
        Returns:
            Classification of the change type
        """
        title = change.get('title', '').lower()
        description = change.get('description', '').lower()
        text = f"{title} {description}"
        
        # Simple keyword-based classification
        # In production, use ML model
        if any(kw in text for kw in ['optimize', 'optimization', 'faster', 'speedup']):
            return ChangeType.OPTIMIZATION
        elif any(kw in text for kw in ['fix', 'bug', 'crash', 'error']):
            return ChangeType.BUG_FIX
        elif any(kw in text for kw in ['add', 'new', 'implement']):
            return ChangeType.NEW_KERNEL
        elif any(kw in text for kw in ['refactor', 'restructure', 'cleanup']):
            return ChangeType.REFACTORING
        else:
            return ChangeType.API_CHANGE
            
    def _determine_priority(self, change: Dict, change_type: ChangeType) -> Priority:
        """
        Determine priority of a change.
        
        Args:
            change: Change metadata
            change_type: Type of change
            
        Returns:
            Priority level
        """
        # High priority for performance optimizations
        if change_type == ChangeType.OPTIMIZATION:
            return Priority.HIGH
            
        # Medium priority for new kernels
        if change_type == ChangeType.NEW_KERNEL:
            return Priority.MEDIUM
            
        # Low priority for refactoring
        if change_type == ChangeType.REFACTORING:
            return Priority.LOW
            
        return Priority.MEDIUM
        
    async def _process_change(self, change: KernelChange):
        """
        Process a detected change.
        
        Args:
            change: Detected kernel change
        """
        logger.info(
            f"Detected {change.change_type.value} in {change.repository}: "
            f"{change.title} (priority: {change.priority.value})"
        )
        
        # In a real implementation:
        # 1. Store change in database
        # 2. Queue for analysis
        # 3. Notify relevant systems
        # 4. Update metrics
        
    async def fetch_change_details(self, change_id: str) -> Optional[KernelChange]:
        """
        Fetch detailed information about a specific change.
        
        Args:
            change_id: Unique identifier for the change
            
        Returns:
            KernelChange object with full details, or None if not found
        """
        # TODO: Implement database query or API call
        logger.info(f"Fetching details for change {change_id}")
        return None
        
    async def get_recent_changes(
        self,
        repository: Optional[str] = None,
        change_type: Optional[ChangeType] = None,
        priority: Optional[Priority] = None,
        limit: int = 100
    ) -> List[KernelChange]:
        """
        Get recently detected changes with optional filters.
        
        Args:
            repository: Filter by repository (format: "owner/repo")
            change_type: Filter by change type
            priority: Filter by priority level
            limit: Maximum number of results
            
        Returns:
            List of kernel changes matching criteria
        """
        # TODO: Implement database query with filters
        logger.info(f"Fetching recent changes (limit: {limit})")
        return []


# Example usage
if __name__ == "__main__":
    async def main():
        # Initialize detector
        detector = ChangeDetector(api_token="your_github_token_here")
        
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
        )
        
        # Start monitoring
        await detector.start_monitoring()
    
    # Run the async main function
    asyncio.run(main())
