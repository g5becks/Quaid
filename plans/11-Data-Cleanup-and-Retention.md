# 11 - Data Cleanup and Retention

**Automated data management for git repository storage using scheduled cleanup tasks**

---

## Executive Summary

Since Quaid stores all data in a git repository, implementing automated data cleanup is essential to prevent indefinite repository growth. This system uses the `schedule` library to run periodic cleanup tasks that remove old data based on configurable retention policies while maintaining data integrity and git history.

**Core Philosophy**: Keep the repository lightweight and performant by automatically cleaning up old data while preserving important knowledge and maintaining git-native storage benefits.

---

## Architecture Overview

### Cleanup System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Data Cleanup System                      │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Configuration Layer                    │   │
│  │                                                     │   │
│  │  • Retention policies (days, counts, types)        │   │
│  │  • Cleanup schedules (daily, weekly, monthly)      │   │
│  │  • Exclusion rules (important fragments)            │   │
│  │  • Cleanup thresholds and limits                    │   │
│  └─────────────────────────────────────────────────────┘   │
│                            ↓                                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Scheduler Engine                        │   │
│  │                                                     │   │
│  │  • Schedule library for periodic tasks             │   │
│  │  • Background task execution                        │   │
│  │  • Task queue management                           │   │
│  │  • Error handling and retry logic                  │   │
│  └─────────────────────────────────────────────────────┘   │
│                            ↓                                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Cleanup Operations                      │   │
│  │                                                     │   │
│  │  • Fragment cleanup (old, low-importance)           │   │
│  │  • Index maintenance (rebuild, optimize)            │   │
│  │  • Cache cleanup (temporary files)                  │   │
│  │  • Git operations (commit, gc, compression)         │   │
│  └─────────────────────────────────────────────────────┘   │
│                            ↓                                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Monitoring & Logging                    │   │
│  │                                                     │   │
│  │  • Cleanup metrics and statistics                  │   │
│  │  • Audit logs of all cleanup operations            │   │
│  │  • Performance monitoring                          │   │
│  │  • Alerting for cleanup issues                     │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## Configuration System

### Cleanup Configuration (config.toml)

```toml
[cleanup]
# Enable automated cleanup
enabled = true

# Global retention settings
default_retention_days = 365
max_fragments_total = 10000
cleanup_schedule = "daily"  # daily, weekly, monthly

[cleanup.fragments]
# Fragment-specific retention policies
by_importance = [
    { importance = "high", retention_days = 730 },    # 2 years
    { importance = "medium", retention_days = 365 },   # 1 year
    { importance = "low", retention_days = 90 }       # 3 months
]

by_type = [
    { type = "concept", retention_days = 730 },       # Keep concepts longer
    { type = "decision", retention_days = 1095 },     # Keep decisions 3 years
    { type = "implementation", retention_days = 180 }, # Code gets outdated faster
    { type = "reference", retention_days = 90 }       # References expire quickly
]

# Minimum counts to preserve
min_counts = [
    { type = "concept", min_count = 100 },
    { type = "decision", min_count = 50 },
    { type = "implementation", min_count = 200 }
]

[cleanup.indexes]
# Index cleanup settings
rebuild_schedule = "weekly"
optimize_schedule = "monthly"
max_index_size_mb = 500

[cleanup.cache]
# Cache cleanup settings
max_cache_size_mb = 100
cache_retention_hours = 24
temp_file_retention_hours = 6

[cleanup.git]
# Git repository maintenance
auto_gc = true
gc_schedule = "monthly"
gc_aggressive = false  # Set to true for monthly deep cleanup
commit_cleanup = true
max_commit_history = 1000

[cleanup.exclusions]
# Never delete these fragments
protected_fragments = [
    ".*-template-.*",     # Template fragments
    ".*-important-.*",    # Marked as important
    "startup-.*",         # Startup knowledge
    "architecture-.*"     # Core architecture
]

protected_tags = [
    "permanent",
    "archival",
    "critical",
    "documentation"
]

[cleanup.monitoring]
# Monitoring and alerting
log_cleanup_operations = true
cleanup_log_file = ".quaid/logs/cleanup.log"
max_log_size_mb = 10
alert_on_failure = true
cleanup_timeout_minutes = 30
```

---

## Core Components

### 1. Cleanup Manager

```python
import schedule
import time
import threading
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
import toml
import polars as pl
import git

@dataclass
class CleanupResult:
    """Results of a cleanup operation"""
    operation: str
    success: bool
    items_processed: int
    items_deleted: int
    space_freed_mb: float
    duration_seconds: float
    error_message: Optional[str] = None
    details: Dict[str, any] = None

@dataclass
class RetentionPolicy:
    """Data retention policy configuration"""
    importance: Optional[str] = None
    fragment_type: Optional[str] = None
    retention_days: int = 365
    min_count: int = 0
    max_age: Optional[datetime] = None

class CleanupManager:
    """Manages automated data cleanup operations"""

    def __init__(self, config_path: str, storage_path: str):
        self.config_path = Path(config_path)
        self.storage_path = Path(storage_path)
        self.logger = self._setup_logging()

        # Load configuration
        self.config = self._load_config()
        self.retention_policies = self._build_retention_policies()

        # Scheduler
        self.scheduler_thread = None
        self.scheduler_running = False

        # Git repository
        self.repo = None
        try:
            self.repo = git.Repo(self.storage_path.parent)
        except git.InvalidGitRepositoryError:
            self.logger.warning(f"Not in a git repository: {self.storage_path.parent}")

    def _setup_logging(self) -> logging.Logger:
        """Setup cleanup-specific logging"""
        logger = logging.getLogger("quaid.cleanup")

        # Create logs directory
        log_dir = Path(".quaid/logs")
        log_dir.mkdir(exist_ok=True)

        # Setup file handler
        handler = logging.FileHandler(log_dir / "cleanup.log")
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        return logger

    def _load_config(self) -> Dict:
        """Load cleanup configuration from TOML file"""
        try:
            config_file = self.config_path
            if config_file.exists():
                return toml.load(config_file).get('cleanup', {})
            else:
                self.logger.warning(f"Cleanup config not found: {config_file}")
                return self._default_config()
        except Exception as e:
            self.logger.error(f"Error loading cleanup config: {e}")
            return self._default_config()

    def _default_config(self) -> Dict:
        """Default cleanup configuration"""
        return {
            'enabled': True,
            'default_retention_days': 365,
            'max_fragments_total': 10000,
            'cleanup_schedule': 'daily',
            'fragments': {
                'by_importance': [
                    {'importance': 'high', 'retention_days': 730},
                    {'importance': 'medium', 'retention_days': 365},
                    {'importance': 'low', 'retention_days': 90}
                ],
                'by_type': [
                    {'type': 'concept', 'retention_days': 730},
                    {'type': 'decision', 'retention_days': 1095},
                    {'type': 'implementation', 'retention_days': 180},
                    {'type': 'reference', 'retention_days': 90}
                ],
                'min_counts': [
                    {'type': 'concept', 'min_count': 100},
                    {'type': 'decision', 'min_count': 50},
                    {'type': 'implementation', 'min_count': 200}
                ]
            },
            'exclusions': {
                'protected_fragments': ['.*-template-.*', '.*-important-.*'],
                'protected_tags': ['permanent', 'archival', 'critical']
            }
        }

    def _build_retention_policies(self) -> List[RetentionPolicy]:
        """Build retention policies from configuration"""
        policies = []

        # Importance-based policies
        for policy_config in self.config.get('fragments', {}).get('by_importance', []):
            policy = RetentionPolicy(
                importance=policy_config['importance'],
                retention_days=policy_config['retention_days']
            )
            policies.append(policy)

        # Type-based policies
        for policy_config in self.config.get('fragments', {}).get('by_type', []):
            policy = RetentionPolicy(
                fragment_type=policy_config['type'],
                retention_days=policy_config['retention_days']
            )
            policies.append(policy)

        # Default policy
        default_days = self.config.get('default_retention_days', 365)
        policies.append(RetentionPolicy(retention_days=default_days))

        return policies

    def start_scheduler(self) -> None:
        """Start the cleanup scheduler in background thread"""
        if self.scheduler_running:
            self.logger.warning("Cleanup scheduler already running")
            return

        if not self.config.get('enabled', False):
            self.logger.info("Cleanup is disabled in configuration")
            return

        self._setup_schedules()

        def run_scheduler():
            self.scheduler_running = True
            self.logger.info("Cleanup scheduler started")

            while self.scheduler_running:
                schedule.run_pending()
                time.sleep(60)  # Check every minute

        self.scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        self.scheduler_thread.start()

    def stop_scheduler(self) -> None:
        """Stop the cleanup scheduler"""
        self.scheduler_running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        self.logger.info("Cleanup scheduler stopped")

    def _setup_schedules(self) -> None:
        """Setup scheduled cleanup tasks"""
        # Clear existing schedules
        schedule.clear()

        cleanup_schedule = self.config.get('cleanup_schedule', 'daily')

        if cleanup_schedule == 'daily':
            schedule.every().day.at("02:00").do(self._run_daily_cleanup)
        elif cleanup_schedule == 'weekly':
            schedule.every().sunday.at("03:00").do(self._run_weekly_cleanup)
        elif cleanup_schedule == 'monthly':
            schedule.every().month.do(self._run_monthly_cleanup)

        # Index maintenance
        schedule.every().week.at("04:00").do(self._maintain_indexes)

        # Git maintenance
        if self.config.get('git', {}).get('auto_gc', True):
            schedule.every().month.do(self._git_maintenance)

        self.logger.info(f"Scheduled cleanup tasks: {cleanup_schedule}")

    def _run_daily_cleanup(self) -> CleanupResult:
        """Run daily cleanup operations"""
        start_time = time.time()
        self.logger.info("Starting daily cleanup")

        try:
            results = []

            # Fragment cleanup
            fragment_result = self.cleanup_fragments()
            results.append(fragment_result)

            # Cache cleanup
            cache_result = self.cleanup_cache()
            results.append(cache_result)

            # Summary
            total_deleted = sum(r.items_deleted for r in results)
            total_space_freed = sum(r.space_freed_mb for r in results)

            # Commit cleanup changes if any
            if total_deleted > 0 and self.repo:
                self._commit_cleanup_changes("daily", total_deleted, total_space_freed)

            duration = time.time() - start_time
            self.logger.info(f"Daily cleanup completed: {total_deleted} items deleted, "
                           f"{total_space_freed:.1f}MB freed in {duration:.1f}s")

            return CleanupResult(
                operation="daily_cleanup",
                success=True,
                items_processed=sum(r.items_processed for r in results),
                items_deleted=total_deleted,
                space_freed_mb=total_space_freed,
                duration_seconds=duration,
                details={"individual_results": [r.__dict__ for r in results]}
            )

        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"Daily cleanup failed: {e}")
            return CleanupResult(
                operation="daily_cleanup",
                success=False,
                items_processed=0,
                items_deleted=0,
                space_freed_mb=0.0,
                duration_seconds=duration,
                error_message=str(e)
            )

    def _run_weekly_cleanup(self) -> CleanupResult:
        """Run weekly cleanup operations"""
        start_time = time.time()
        self.logger.info("Starting weekly cleanup")

        try:
            results = []

            # Run daily cleanup first
            daily_result = self._run_daily_cleanup()
            results.append(daily_result)

            # Additional weekly operations
            index_result = self._optimize_indexes()
            results.append(index_result)

            total_deleted = sum(r.items_deleted for r in results)
            total_space_freed = sum(r.space_freed_mb for r in results)

            if total_deleted > 0 and self.repo:
                self._commit_cleanup_changes("weekly", total_deleted, total_space_freed)

            duration = time.time() - start_time
            self.logger.info(f"Weekly cleanup completed: {total_deleted} items deleted, "
                           f"{total_space_freed:.1f}MB freed in {duration:.1f}s")

            return CleanupResult(
                operation="weekly_cleanup",
                success=True,
                items_processed=sum(r.items_processed for r in results),
                items_deleted=total_deleted,
                space_freed_mb=total_space_freed,
                duration_seconds=duration
            )

        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"Weekly cleanup failed: {e}")
            return CleanupResult(
                operation="weekly_cleanup",
                success=False,
                items_processed=0,
                items_deleted=0,
                space_freed_mb=0.0,
                duration_seconds=duration,
                error_message=str(e)
            )

    def _run_monthly_cleanup(self) -> CleanupResult:
        """Run monthly deep cleanup operations"""
        start_time = time.time()
        self.logger.info("Starting monthly cleanup")

        try:
            results = []

            # Run weekly cleanup first
            weekly_result = self._run_weekly_cleanup()
            results.append(weekly_result)

            # Deep monthly operations
            if self.config.get('git', {}).get('gc_aggressive', False):
                git_result = self._aggressive_git_cleanup()
                results.append(git_result)

            total_deleted = sum(r.items_deleted for r in results)
            total_space_freed = sum(r.space_freed_mb for r in results)

            if total_deleted > 0 and self.repo:
                self._commit_cleanup_changes("monthly", total_deleted, total_space_freed)

            duration = time.time() - start_time
            self.logger.info(f"Monthly cleanup completed: {total_deleted} items deleted, "
                           f"{total_space_freed:.1f}MB freed in {duration:.1f}s")

            return CleanupResult(
                operation="monthly_cleanup",
                success=True,
                items_processed=sum(r.items_processed for r in results),
                items_deleted=total_deleted,
                space_freed_mb=total_space_freed,
                duration_seconds=duration
            )

        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"Monthly cleanup failed: {e}")
            return CleanupResult(
                operation="monthly_cleanup",
                success=False,
                items_processed=0,
                items_deleted=0,
                space_freed_mb=0.0,
                duration_seconds=duration,
                error_message=str(e)
            )
```

### 2. Fragment Cleanup Operations

```python
import re
from datetime import datetime
import polars as pl

class FragmentCleaner:
    """Handles fragment-specific cleanup operations"""

    def __init__(self, storage_path: str, retention_policies: List[RetentionPolicy],
                 exclusions: Dict):
        self.storage_path = Path(storage_path)
        self.retention_policies = retention_policies
        self.exclusions = exclusions
        self.logger = logging.getLogger("quaid.cleanup.fragments")

    def cleanup_fragments(self) -> CleanupResult:
        """Main fragment cleanup operation"""
        start_time = time.time()

        try:
            # Load fragments index
            fragments_df = self._load_fragments_index()
            if fragments_df is None or len(fragments_df) == 0:
                return CleanupResult(
                    operation="fragment_cleanup",
                    success=True,
                    items_processed=0,
                    items_deleted=0,
                    space_freed_mb=0.0,
                    duration_seconds=time.time() - start_time
                )

            # Identify fragments to delete
            fragments_to_delete = self._identify_fragments_for_cleanup(fragments_df)

            # Apply minimum count constraints
            fragments_to_delete = self._apply_minimum_counts(fragments_to_delete, fragments_df)

            # Apply exclusions
            fragments_to_delete = self._apply_exclusions(fragments_to_delete, fragments_df)

            # Delete fragments
            deleted_count, space_freed = self._delete_fragments(fragments_to_delete)

            # Update indexes
            self._update_indexes(fragments_to_delete)

            duration = time.time() - start_time
            self.logger.info(f"Fragment cleanup: {deleted_count} fragments deleted, "
                           f"{space_freed:.1f}MB freed in {duration:.1f}s")

            return CleanupResult(
                operation="fragment_cleanup",
                success=True,
                items_processed=len(fragments_df),
                items_deleted=deleted_count,
                space_freed_mb=space_freed,
                duration_seconds=duration,
                details={
                    "fragments_considered": len(fragments_df),
                    "fragments_marked_for_deletion": len(fragments_to_delete),
                    "deletion_criteria_applied": ["retention_age", "minimum_counts", "exclusions"]
                }
            )

        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"Fragment cleanup failed: {e}")
            return CleanupResult(
                operation="fragment_cleanup",
                success=False,
                items_processed=0,
                items_deleted=0,
                space_freed_mb=0.0,
                duration_seconds=duration,
                error_message=str(e)
            )

    def _load_fragments_index(self) -> Optional[pl.DataFrame]:
        """Load fragments index from JSONL"""
        try:
            fragments_file = self.storage_path / "indexes" / "fragments.jsonl"
            if fragments_file.exists():
                return pl.read_ndjson(fragments_file)
            return None
        except Exception as e:
            self.logger.error(f"Error loading fragments index: {e}")
            return None

    def _identify_fragments_for_cleanup(self, fragments_df: pl.DataFrame) -> List[str]:
        """Identify fragments that meet cleanup criteria based on retention policies"""
        fragments_to_delete = []
        cutoff_date = datetime.now()

        for fragment in fragments_df.iter_rows(named=True):
            should_delete = False

            # Check each retention policy
            for policy in self.retention_policies:
                if self._fragment_matches_policy(fragment, policy):
                    policy_cutoff = cutoff_date - timedelta(days=policy.retention_days)
                    fragment_created = datetime.fromisoformat(fragment['created'])

                    if fragment_created < policy_cutoff:
                        should_delete = True
                        break

            if should_delete:
                fragments_to_delete.append(fragment['id'])

        return fragments_to_delete

    def _fragment_matches_policy(self, fragment: Dict, policy: RetentionPolicy) -> bool:
        """Check if fragment matches a retention policy"""
        # Check importance match
        if policy.importance and fragment.get('importance') != policy.importance:
            return False

        # Check type match
        if policy.fragment_type and fragment.get('type') != policy.fragment_type:
            return False

        # Default policy matches all
        if not policy.importance and not policy.fragment_type:
            return True

        return True

    def _apply_minimum_counts(self, fragments_to_delete: List[str],
                            fragments_df: pl.DataFrame) -> List[str]:
        """Apply minimum count constraints to preserve important fragments"""
        try:
            min_counts = self.exclusions.get('min_counts', {})
            remaining_fragments = []

            for fragment_id in fragments_to_delete:
                should_delete = True

                # Find fragment data
                fragment_data = fragments_df.filter(pl.col("id") == fragment_id)
                if len(fragment_data) == 0:
                    continue

                fragment_type = fragment_data['type'][0]

                # Check minimum count for this type
                if fragment_type in min_counts:
                    min_count = min_counts[fragment_type]

                    # Count remaining fragments of this type
                    current_count = len(fragments_df.filter(
                        (pl.col("type") == fragment_type) &
                        (~pl.col("id").is_in(fragments_to_delete)) |
                        (pl.col("id") == fragment_id)
                    ))

                    if current_count <= min_count:
                        should_delete = False
                        self.logger.debug(f"Preserving {fragment_id} to meet minimum count "
                                        f"for type {fragment_type}")

                if should_delete:
                    remaining_fragments.append(fragment_id)

            return remaining_fragments

        except Exception as e:
            self.logger.error(f"Error applying minimum counts: {e}")
            return fragments_to_delete

    def _apply_exclusions(self, fragments_to_delete: List[str],
                        fragments_df: pl.DataFrame) -> List[str]:
        """Apply exclusion rules to protected fragments"""
        try:
            protected_patterns = self.exclusions.get('protected_fragments', [])
            protected_tags = self.exclusions.get('protected_tags', [])
            remaining_fragments = []

            for fragment_id in fragments_to_delete:
                should_delete = True

                # Find fragment data
                fragment_data = fragments_df.filter(pl.col("id") == fragment_id)
                if len(fragment_data) == 0:
                    continue

                fragment = fragment_data.to_dicts()[0]

                # Check protected patterns
                for pattern in protected_patterns:
                    if re.match(pattern, fragment_id):
                        should_delete = False
                        self.logger.debug(f"Preserving {fragment_id} (matches protected pattern: {pattern})")
                        break

                # Check protected tags
                if should_delete and fragment.get('tags'):
                    fragment_tags = fragment['tags']
                    for protected_tag in protected_tags:
                        if protected_tag in fragment_tags:
                            should_delete = False
                            self.logger.debug(f"Preserving {fragment_id} (has protected tag: {protected_tag})")
                            break

                if should_delete:
                    remaining_fragments.append(fragment_id)

            return remaining_fragments

        except Exception as e:
            self.logger.error(f"Error applying exclusions: {e}")
            return fragments_to_delete

    def _delete_fragments(self, fragment_ids: List[str]) -> Tuple[int, float]:
        """Delete fragment files and calculate space freed"""
        deleted_count = 0
        space_freed = 0.0

        for fragment_id in fragment_ids:
            try:
                # Find fragment file
                fragment_file = None
                fragments_dir = self.storage_path / "fragments"

                if fragments_dir.exists():
                    for file_path in fragments_dir.glob("*.md"):
                        if file_path.stem == fragment_id:
                            fragment_file = file_path
                            break

                if fragment_file and fragment_file.exists():
                    # Calculate file size
                    file_size = fragment_file.stat().st_size / (1024 * 1024)  # MB

                    # Delete file
                    fragment_file.unlink()
                    deleted_count += 1
                    space_freed += file_size

                    self.logger.debug(f"Deleted fragment: {fragment_id} ({file_size:.1f}MB)")
                else:
                    self.logger.warning(f"Fragment file not found: {fragment_id}")

            except Exception as e:
                self.logger.error(f"Error deleting fragment {fragment_id}: {e}")

        return deleted_count, space_freed

    def _update_indexes(self, deleted_fragment_ids: List[str]) -> None:
        """Update indexes after fragment deletion"""
        try:
            # Update fragments index
            fragments_file = self.storage_path / "indexes" / "fragments.jsonl"
            if fragments_file.exists():
                fragments_df = pl.read_ndjson(fragments_file)
                updated_df = fragments_df.filter(~pl.col("id").is_in(deleted_fragment_ids))
                updated_df.write_ndjson(fragments_file)

            # Update graph index
            graph_file = self.storage_path / "indexes" / "graph.jsonl"
            if graph_file.exists():
                graph_df = pl.read_ndjson(graph_file)
                updated_df = graph_df.filter(
                    (~pl.col("from_id").is_in(deleted_fragment_ids)) &
                    (~pl.col("to_id").is_in(deleted_fragment_ids))
                )
                updated_df.write_ndjson(graph_file)

            self.logger.info(f"Updated indexes for {len(deleted_fragment_ids)} deleted fragments")

        except Exception as e:
            self.logger.error(f"Error updating indexes: {e}")
```

### 3. Cache and Index Cleanup

```python
import shutil
import glob

class SystemCleaner:
    """Handles cache and system cleanup operations"""

    def __init__(self, storage_path: str, config: Dict):
        self.storage_path = Path(storage_path)
        self.config = config
        self.logger = logging.getLogger("quaid.cleanup.system")

    def cleanup_cache(self) -> CleanupResult:
        """Clean up cache and temporary files"""
        start_time = time.time()

        try:
            deleted_items = 0
            space_freed = 0.0

            # Clean cache directory
            cache_dir = self.storage_path / "cache"
            if cache_dir.exists():
                cache_result = self._clean_directory(cache_dir, hours=24)
                deleted_items += cache_result[0]
                space_freed += cache_result[1]

            # Clean temporary files
            temp_patterns = [
                self.storage_path / "*.tmp",
                self.storage_path / "*.temp",
                self.storage_path / "**/*.tmp",
                self.storage_path / "**/*.temp"
            ]

            for pattern in temp_patterns:
                temp_result = self._clean_glob_pattern(pattern, hours=6)
                deleted_items += temp_result[0]
                space_freed += temp_result[1]

            # Clean log files if they exceed size limit
            log_result = self._clean_logs()
            deleted_items += log_result[0]
            space_freed += log_result[1]

            duration = time.time() - start_time
            self.logger.info(f"Cache cleanup: {deleted_items} files deleted, "
                           f"{space_freed:.1f}MB freed in {duration:.1f}s")

            return CleanupResult(
                operation="cache_cleanup",
                success=True,
                items_processed=deleted_items,
                items_deleted=deleted_items,
                space_freed_mb=space_freed,
                duration_seconds=duration
            )

        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"Cache cleanup failed: {e}")
            return CleanupResult(
                operation="cache_cleanup",
                success=False,
                items_processed=0,
                items_deleted=0,
                space_freed_mb=0.0,
                duration_seconds=duration,
                error_message=str(e)
            )

    def _clean_directory(self, directory: Path, hours: int) -> Tuple[int, float]:
        """Clean files in directory older than specified hours"""
        deleted_count = 0
        space_freed = 0.0
        cutoff_time = time.time() - (hours * 3600)

        try:
            for file_path in directory.rglob("*"):
                if file_path.is_file():
                    if file_path.stat().st_mtime < cutoff_time:
                        file_size = file_path.stat().st_size / (1024 * 1024)  # MB
                        file_path.unlink()
                        deleted_count += 1
                        space_freed += file_size
        except Exception as e:
            self.logger.error(f"Error cleaning directory {directory}: {e}")

        return deleted_count, space_freed

    def _clean_glob_pattern(self, pattern: Path, hours: int) -> Tuple[int, float]:
        """Clean files matching glob pattern older than specified hours"""
        deleted_count = 0
        space_freed = 0.0
        cutoff_time = time.time() - (hours * 3600)

        try:
            for file_path in glob.glob(str(pattern)):
                file_path = Path(file_path)
                if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                    file_size = file_path.stat().st_size / (1024 * 1024)  # MB
                    file_path.unlink()
                    deleted_count += 1
                    space_freed += file_size
        except Exception as e:
            self.logger.error(f"Error cleaning pattern {pattern}: {e}")

        return deleted_count, space_freed

    def _clean_logs(self) -> Tuple[int, float]:
        """Clean log files that exceed size limits"""
        deleted_count = 0
        space_freed = 0.0

        try:
            log_dir = self.storage_path / "logs"
            if not log_dir.exists():
                return 0, 0.0

            max_log_size = self.config.get('monitoring', {}).get('max_log_size_mb', 10) * 1024 * 1024

            for log_file in log_dir.glob("*.log"):
                if log_file.stat().st_size > max_log_size:
                    # Rotate log file
                    backup_file = log_file.with_suffix(f".{int(time.time())}.bak")
                    log_file.rename(backup_file)

                    # Keep only last 5 backup files
                    backups = sorted(log_dir.glob(f"{log_file.stem}.*.bak"))
                    if len(backups) > 5:
                        for old_backup in backups[:-5]:
                            file_size = old_backup.stat().st_size / (1024 * 1024)
                            old_backup.unlink()
                            deleted_count += 1
                            space_freed += file_size

        except Exception as e:
            self.logger.error(f"Error cleaning logs: {e}")

        return deleted_count, space_freed

    def _optimize_indexes(self) -> CleanupResult:
        """Optimize and rebuild indexes"""
        start_time = time.time()

        try:
            optimized_count = 0
            space_freed = 0.0

            # Rebuild Tantivy index if it exists
            tantivy_dir = self.storage_path / "indexes" / "tantivy"
            if tantivy_dir.exists():
                # Get size before optimization
                old_size = sum(f.stat().st_size for f in tantivy_dir.rglob("*") if f.is_file())

                # Tantivy optimization would go here
                # For now, we'll just log it
                self.logger.info("Tantivy index optimization not implemented yet")

                # Get size after optimization
                new_size = sum(f.stat().st_size for f in tantivy_dir.rglob("*") if f.is_file())
                space_freed = (old_size - new_size) / (1024 * 1024)
                optimized_count += 1

            duration = time.time() - start_time
            self.logger.info(f"Index optimization: {optimized_count} indexes optimized, "
                           f"{space_freed:.1f}MB freed in {duration:.1f}s")

            return CleanupResult(
                operation="index_optimization",
                success=True,
                items_processed=optimized_count,
                items_deleted=0,  # Optimization doesn't delete, just compresses
                space_freed_mb=space_freed,
                duration_seconds=duration
            )

        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"Index optimization failed: {e}")
            return CleanupResult(
                operation="index_optimization",
                success=False,
                items_processed=0,
                items_deleted=0,
                space_freed_mb=0.0,
                duration_seconds=duration,
                error_message=str(e)
            )
```

### 4. Git Repository Maintenance

```python
class GitMaintenance:
    """Handles git repository maintenance operations"""

    def __init__(self, repo: git.Repo, config: Dict):
        self.repo = repo
        self.config = config
        self.logger = logging.getLogger("quaid.cleanup.git")

    def _commit_cleanup_changes(self, cleanup_type: str, deleted_count: int, space_freed_mb: float) -> None:
        """Commit cleanup changes to git"""
        try:
            if not self.repo:
                return

            # Check if there are changes to commit
            if not self.repo.is_dirty(untracked_files=True):
                return

            # Stage all changes
            self.repo.git.add(A=True)

            # Create commit message
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            commit_message = f""" Automated {cleanup_type} cleanup

- Deleted {deleted_count} fragments
- Freed {space_freed_mb:.1f}MB of storage
- Completed at {timestamp}

Generated by Quaid cleanup system
"""

            # Commit changes
            self.repo.index.commit(commit_message)
            self.logger.info(f"Committed {cleanup_type} cleanup changes")

        except Exception as e:
            self.logger.error(f"Error committing cleanup changes: {e}")

    def _git_maintenance(self) -> CleanupResult:
        """Perform git repository maintenance"""
        start_time = time.time()

        try:
            if not self.repo:
                return CleanupResult(
                    operation="git_maintenance",
                    success=False,
                    items_processed=0,
                    items_deleted=0,
                    space_freed_mb=0.0,
                    duration_seconds=0,
                    error_message="Not in a git repository"
                )

            # Get repository size before maintenance
            old_size = self._get_repo_size()

            # Run git garbage collection
            if self.config.get('gc_aggressive', False):
                self.repo.git.gc('--aggressive', '--prune=now')
            else:
                self.repo.git.gc('--prune=now')

            # Clean up unreachable commits
            if self.config.get('commit_cleanup', True):
                max_commits = self.config.get('max_commit_history', 1000)
                self._cleanup_commit_history(max_commits)

            # Get repository size after maintenance
            new_size = self._get_repo_size()
            space_freed = (old_size - new_size) / (1024 * 1024)

            duration = time.time() - start_time
            self.logger.info(f"Git maintenance: {space_freed:.1f}MB freed in {duration:.1f}s")

            return CleanupResult(
                operation="git_maintenance",
                success=True,
                items_processed=1,  # Repository
                items_deleted=0,
                space_freed_mb=space_freed,
                duration_seconds=duration
            )

        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"Git maintenance failed: {e}")
            return CleanupResult(
                operation="git_maintenance",
                success=False,
                items_processed=0,
                items_deleted=0,
                space_freed_mb=0.0,
                duration_seconds=duration,
                error_message=str(e)
            )

    def _get_repo_size(self) -> int:
        """Get total repository size in bytes"""
        total_size = 0
        for file_path in self.repo.working_dir.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return total_size

    def _cleanup_commit_history(self, max_commits: int) -> None:
        """Clean up old commit history while preserving important commits"""
        try:
            # Get all commits
            commits = list(self.repo.iter_commits())

            if len(commits) <= max_commits:
                return

            # Identify commits to keep (cleanup commits are less important)
            commits_to_keep = []
            commits_to_remove = []

            for commit in commits:
                commit_message = commit.message.lower()

                # Keep important commits
                if any(keyword in commit_message for keyword in [
                    'initial', 'setup', 'architecture', 'migration', 'merge', 'release'
                ]):
                    commits_to_keep.append(commit)
                # Mark cleanup commits for removal
                elif 'cleanup' in commit_message:
                    commits_to_remove.append(commit)
                # Keep others until we reach the limit
                elif len(commits_to_keep) < max_commits:
                    commits_to_keep.append(commit)
                else:
                    commits_to_remove.append(commit)

            if commits_to_remove:
                self.logger.info(f"Identified {len(commits_to_remove)} commits for potential cleanup")
                # Actual commit cleanup would require careful implementation
                # For now, we'll just log it

        except Exception as e:
            self.logger.error(f"Error cleaning up commit history: {e}")
```

---

## MCP Server Integration

### Cleanup Tools for MCP Server

```python
@mcp.tool()
async def cleanup_run_manual(cleanup_type: str = "daily", dry_run: bool = True) -> dict:
    """Run manual cleanup operation"""
    try:
        cleanup_manager = CleanupManager(".quaid/config.toml", ".quaid/memory")

        if dry_run:
            # Dry run - just report what would be deleted
            fragments_to_delete = cleanup_manager._identify_fragments_for_deletion()
            return {
                "dry_run": True,
                "cleanup_type": cleanup_type,
                "fragments_to_delete": len(fragments_to_delete),
                "estimated_space_freed_mb": len(fragments_to_delete) * 0.5,  # Rough estimate
                "fragments": fragments_to_delete[:10]  # First 10 as sample
            }
        else:
            # Actual cleanup
            if cleanup_type == "daily":
                result = cleanup_manager._run_daily_cleanup()
            elif cleanup_type == "weekly":
                result = cleanup_manager._run_weekly_cleanup()
            elif cleanup_type == "monthly":
                result = cleanup_manager._run_monthly_cleanup()
            else:
                return {"error": f"Invalid cleanup type: {cleanup_type}"}

            return {
                "success": result.success,
                "cleanup_type": cleanup_type,
                "items_processed": result.items_processed,
                "items_deleted": result.items_deleted,
                "space_freed_mb": result.space_freed_mb,
                "duration_seconds": result.duration_seconds,
                "error_message": result.error_message
            }

    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
async def cleanup_show_retention_policies() -> dict:
    """Show current retention policies"""
    try:
        cleanup_manager = CleanupManager(".quaid/config.toml", ".quaid/memory")

        policies = []
        for policy in cleanup_manager.retention_policies:
            policy_dict = {
                "importance": policy.importance,
                "fragment_type": policy.fragment_type,
                "retention_days": policy.retention_days
            }
            policies.append(policy_dict)

        return {
            "retention_policies": policies,
            "default_retention_days": cleanup_manager.config.get('default_retention_days', 365),
            "max_fragments_total": cleanup_manager.config.get('max_fragments_total', 10000),
            "cleanup_schedule": cleanup_manager.config.get('cleanup_schedule', 'daily'),
            "protected_patterns": cleanup_manager.config.get('exclusions', {}).get('protected_fragments', []),
            "protected_tags": cleanup_manager.config.get('exclusions', {}).get('protected_tags', [])
        }

    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
async def cleanup_get_statistics() -> dict:
    """Get cleanup statistics and repository information"""
    try:
        storage_path = Path(".quaid/memory")

        # Count fragments
        fragments_dir = storage_path / "fragments"
        fragment_count = len(list(fragments_dir.glob("*.md"))) if fragments_dir.exists() else 0

        # Calculate repository size
        repo_size = 0
        if storage_path.exists():
            for file_path in storage_path.rglob("*"):
                if file_path.is_file():
                    repo_size += file_path.stat().st_size

        # Get oldest and newest fragments
        oldest_date = None
        newest_date = None

        fragments_file = storage_path / "indexes" / "fragments.jsonl"
        if fragments_file.exists():
            fragments_df = pl.read_ndjson(fragments_file)
            if len(fragments_df) > 0:
                dates = fragments_df['created'].to_list()
                oldest_date = min(dates)
                newest_date = max(dates)

        return {
            "fragment_count": fragment_count,
            "repository_size_mb": repo_size / (1024 * 1024),
            "oldest_fragment": oldest_date,
            "newest_fragment": newest_date,
            "storage_path": str(storage_path),
            "last_cleanup": "Unknown",  # This would be tracked in actual implementation
            "next_scheduled_cleanup": "Daily at 02:00"
        }

    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
async def_cleanup_configure_policy(policy_type: str, importance: Optional[str] = None,
                                fragment_type: Optional[str] = None,
                                retention_days: int = 365) -> dict:
    """Configure retention policy"""
    try:
        # This would update the configuration file
        # For now, return a success message
        return {
            "success": True,
            "message": f"Updated {policy_type} retention policy",
            "policy": {
                "importance": importance,
                "fragment_type": fragment_type,
                "retention_days": retention_days
            },
            "note": "Configuration file update would be implemented here"
        }

    except Exception as e:
        return {"error": str(e)}
```

---

## Usage Examples

### Basic Setup

```python
# Initialize cleanup manager
cleanup_manager = CleanupManager(
    config_path=".quaid/config.toml",
    storage_path=".quaid/memory"
)

# Start scheduler in background
cleanup_manager.start_scheduler()

# Server runs...
# Cleanup tasks run automatically based on schedule

# Stop scheduler when done
cleanup_manager.stop_scheduler()
```

### Manual Cleanup

```python
# Run daily cleanup manually
result = cleanup_manager._run_daily_cleanup()
print(f"Deleted {result.items_deleted} items, freed {result.space_freed_mb:.1f}MB")

# Run dry run to see what would be deleted
dry_run_result = cleanup_manager._run_daily_cleanup()
# In dry run mode, this would only report without deleting
```

### MCP Server Usage

```python
# Check retention policies
policies = await mcp.call("cleanup_show_retention_policies")

# Get repository statistics
stats = await mcp.call("cleanup_get_statistics")

# Run manual cleanup
result = await mcp.call("cleanup_run_manual", cleanup_type="daily", dry_run=True)
```

---

## Dependencies

Add to `pyproject.toml`:

```toml
[project.dependencies]
# ... existing dependencies ...
schedule = "^1.2"                     # Job scheduling
GitPython = "^3.1"                    # Git operations

[project.optional-dependencies]
cleanup = [
    "schedule>=1.2",                   # Job scheduling
    "GitPython>=3.1"                   # Git operations
]

# Update all group
all = [
    "quaid[dev,docs,graph,ai,cleanup]"
]
```

---

## Benefits of Automated Cleanup

1. **Repository Health**: Prevents indefinite growth and keeps repository performant
2. **Storage Efficiency**: Automatically removes outdated data while preserving important knowledge
3. **Configurable Policies**: Flexible retention policies based on importance, type, and age
4. **Safe Operations**: Dry-run mode, exclusions, and minimum count constraints
5. **Git Integration**: Maintains git history while cleaning up data
6. **Monitoring**: Comprehensive logging and metrics for cleanup operations
7. **Scheduled Automation**: Hands-off operation with configurable schedules

---

## Implementation Roadmap

### Phase 1: Core Cleanup Engine (Week 1)
- [ ] Install schedule and GitPython dependencies
- [ ] Implement CleanupManager class
- [ ] Add configuration loading and validation
- [ ] Create basic retention policy system

### Phase 2: Fragment Cleanup (Week 2)
- [ ] Implement FragmentCleaner with policy matching
- [ ] Add exclusion rules and minimum count constraints
- [ ] Create safe deletion with index updates
- [ ] Add comprehensive error handling

### Phase 3: System Maintenance (Week 3)
- [ ] Implement SystemCleaner for cache and temp files
- [ ] Add GitMaintenance for repository cleanup
- [ ] Create index optimization routines
- [ ] Add logging and monitoring

### Phase 4: Scheduler Integration (Week 4)
- [ ] Integrate schedule library for automated tasks
- [ ] Add background thread execution
- [ ] Implement daily/weekly/monthly cleanup routines
- [ ] Add configuration-driven scheduling

### Phase 5: MCP Server Tools (Week 5)
- [ ] Register cleanup tools with MCP server
- [ ] Add manual cleanup and dry-run capabilities
- [ ] Create statistics and monitoring tools
- [ ] Add policy configuration endpoints

### Phase 6: Testing and Monitoring (Week 6)
- [ ] Add comprehensive test coverage
- [ ] Implement cleanup metrics and alerting
- [ ] Add performance monitoring
- [ ] Create backup and rollback procedures

---

**Previous**: [10-NetworkX-Graph-Integration.md](10-NetworkX-Graph-Integration.md) | **Next**: [12-MCP-Server-Reference.md](12-MCP-Server-Reference.md)