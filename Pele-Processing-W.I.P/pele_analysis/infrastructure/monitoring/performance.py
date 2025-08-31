"""
Performance Profiling and Timing

Professional performance monitoring system for identifying bottlenecks
and optimizing computational workflows.
"""

import time
import cProfile
import pstats
import io
from typing import Dict, List, Optional, Any, Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from collections import defaultdict

from ..logging.mpi_logger import get_logger


@dataclass
class TimingData:
    """Timing data for a specific operation"""
    name: str
    start_time: float
    end_time: float
    duration: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class TimingContext:
    """
    Context manager for timing operations

    Provides clean, accurate timing with automatic logging and statistics.
    """

    def __init__(self, name: str, profiler: 'PerformanceProfiler' = None,
                 metadata: Optional[Dict] = None):
        self.name = name
        self.profiler = profiler
        self.metadata = metadata or {}
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        duration = self.end_time - self.start_time

        if self.profiler:
            timing_data = TimingData(
                name=self.name,
                start_time=self.start_time,
                end_time=self.end_time,
                duration=duration,
                metadata=self.metadata
            )
            self.profiler.record_timing(timing_data)

    def get_duration(self) -> Optional[float]:
        """Get duration if timing is complete"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None


class PerformanceProfiler:
    """
    Professional performance profiler for scientific computing

    Provides comprehensive timing analysis, bottleneck identification,
    and performance optimization guidance.
    """

    def __init__(self, enable_detailed_profiling: bool = False):
        self.enable_detailed_profiling = enable_detailed_profiling
        self.logger = get_logger()

        # Timing data storage
        self.timing_data: List[TimingData] = []
        self.operation_stats: Dict[str, List[float]] = defaultdict(list)

        # Profiling data
        self.profiler: Optional[cProfile.Profile] = None
        if enable_detailed_profiling:
            self.profiler = cProfile.Profile()

        # Performance metrics
        self.start_time = time.perf_counter()
        self.total_operations = 0

    def time_operation(self, name: str, metadata: Optional[Dict] = None) -> TimingContext:
        """
        Create timing context for an operation

        Args:
            name: Operation name
            metadata: Optional metadata to store with timing

        Returns:
            TimingContext for use with 'with' statement
        """
        return TimingContext(name, self, metadata)

    def record_timing(self, timing_data: TimingData):
        """Record timing data from completed operation"""

        self.timing_data.append(timing_data)
        self.operation_stats[timing_data.name].append(timing_data.duration)
        self.total_operations += 1

        # Log slow operations
        if timing_data.duration > 60:  # More than 1 minute
            self.logger.warning(f"Slow operation '{timing_data.name}': {timing_data.duration:.2f}s")

    @contextmanager
    def profile_section(self, section_name: str):
        """Profile a section of code with detailed profiling"""

        if not self.enable_detailed_profiling or not self.profiler:
            # Just time the section without detailed profiling
            with self.time_operation(section_name):
                yield
            return

        self.logger.debug(f"Starting detailed profiling of {section_name}")

        # Start detailed profiling
        self.profiler.enable()
        start_time = time.perf_counter()

        try:
            yield
        finally:
            # Stop profiling
            end_time = time.perf_counter()
            self.profiler.disable()

            # Record timing
            timing_data = TimingData(
                name=section_name,
                start_time=start_time,
                end_time=end_time,
                duration=end_time - start_time
            )
            self.record_timing(timing_data)

    def get_operation_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get comprehensive statistics for all operations"""

        stats = {}

        for operation, durations in self.operation_stats.items():
            if not durations:
                continue

            stats[operation] = {
                'count': len(durations),
                'total_time': sum(durations),
                'average_time': sum(durations) / len(durations),
                'min_time': min(durations),
                'max_time': max(durations),
                'median_time': sorted(durations)[len(durations) // 2],
            }

            # Calculate percentiles
            sorted_durations = sorted(durations)
            n = len(sorted_durations)
            stats[operation].update({
                'p95_time': sorted_durations[int(0.95 * n)],
                'p99_time': sorted_durations[int(0.99 * n)],
            })

        return stats

    def identify_bottlenecks(self, top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Identify the top bottlenecks in the processing

        Args:
            top_n: Number of top bottlenecks to return

        Returns:
            List of bottleneck information dictionaries
        """

        operation_stats = self.get_operation_statistics()

        # Sort by total time spent
        bottlenecks = []
        for operation, stats in operation_stats.items():
            bottlenecks.append({
                'operation': operation,
                'total_time': stats['total_time'],
                'average_time': stats['average_time'],
                'count': stats['count'],
                'percentage_of_total': (stats['total_time'] / self.get_total_elapsed_time()) * 100
            })

        # Sort by total time and return top N
        bottlenecks.sort(key=lambda x: x['total_time'], reverse=True)
        return bottlenecks[:top_n]

    def get_detailed_profile_report(self, sort_by: str = 'tottime',
                                    top_n: int = 20) -> str:
        """
        Get detailed profiling report

        Args:
            sort_by: Sort criterion ('tottime', 'cumtime', 'calls')
            top_n: Number of top functions to include

        Returns:
            Formatted profiling report
        """

        if not self.profiler:
            return "Detailed profiling not enabled"

        # Create string buffer for report
        report_buffer = io.StringIO()

        # Generate stats
        stats = pstats.Stats(self.profiler, stream=report_buffer)
        stats.strip_dirs()
        stats.sort_stats(sort_by)
        stats.print_stats(top_n)

        return report_buffer.getvalue()

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""

        total_elapsed = self.get_total_elapsed_time()
        operation_stats = self.get_operation_statistics()
        bottlenecks = self.identify_bottlenecks(top_n=3)

        summary = {
            'total_elapsed_time': total_elapsed,
            'total_operations': self.total_operations,
            'operations_per_second': self.total_operations / total_elapsed if total_elapsed > 0 else 0,
            'unique_operations': len(operation_stats),
            'top_bottlenecks': bottlenecks,
            'detailed_profiling_enabled': self.enable_detailed_profiling
        }

        # Add operation breakdown
        if operation_stats:
            total_operation_time = sum(stats['total_time'] for stats in operation_stats.values())
            summary['total_operation_time'] = total_operation_time
            summary['overhead_percentage'] = ((
                                                          total_elapsed - total_operation_time) / total_elapsed) * 100 if total_elapsed > 0 else 0

        return summary

    def get_total_elapsed_time(self) -> float:
        """Get total elapsed time since profiler creation"""
        return time.perf_counter() - self.start_time

    def generate_performance_report(self) -> str:
        """Generate comprehensive performance report"""

        summary = self.get_performance_summary()
        report_lines = [
            "=" * 60,
            "PERFORMANCE ANALYSIS REPORT",
            "=" * 60,
            f"Total Processing Time: {summary['total_elapsed_time']:.2f} seconds",
            f"Total Operations: {summary['total_operations']}",
            f"Operations per Second: {summary['operations_per_second']:.2f}",
            f"Unique Operation Types: {summary['unique_operations']}",
            "",
            "TOP BOTTLENECKS:",
            "-" * 40
        ]

        for i, bottleneck in enumerate(summary['top_bottlenecks'], 1):
            report_lines.extend([
                f"{i}. {bottleneck['operation']}",
                f"   Total Time: {bottleneck['total_time']:.2f}s ({bottleneck['percentage_of_total']:.1f}%)",
                f"   Average Time: {bottleneck['average_time']:.4f}s",
                f"   Call Count: {bottleneck['count']}",
                ""
            ])

        # Add detailed profiling report if available
        if self.enable_detailed_profiling:
            report_lines.extend([
                "DETAILED FUNCTION PROFILING:",
                "-" * 40,
                self.get_detailed_profile_report(),
            ])

        return "\n".join(report_lines)

    def save_report(self, filename: str):
        """Save performance report to file"""

        report = self.generate_performance_report()

        with open(filename, 'w') as f:
            f.write(report)

        self.logger.info(f"Performance report saved to {filename}")
