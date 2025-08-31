"""
Memory and Resource Monitoring

Professional memory monitoring system for tracking resource usage
and preventing memory leaks in long-running parallel computations.
"""

import psutil
import time
import threading
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime

from ..logging.mpi_logger import get_logger


@dataclass
class MemorySnapshot:
    """Snapshot of memory usage at a point in time"""
    timestamp: float
    virtual_memory_mb: float
    resident_memory_mb: float
    peak_memory_mb: float
    available_memory_mb: float
    memory_percent: float
    swap_memory_mb: float = 0.0


@dataclass
class ResourceSnapshot:
    """Complete resource usage snapshot"""
    timestamp: float
    memory: MemorySnapshot
    cpu_percent: float
    disk_io_read_mb: float = 0.0
    disk_io_write_mb: float = 0.0
    network_io_sent_mb: float = 0.0
    network_io_recv_mb: float = 0.0


class MemoryMonitor:
    """
    Professional memory monitoring system

    Tracks memory usage, detects leaks, and provides cleanup triggers
    for long-running scientific computations.
    """

    def __init__(self, max_memory_gb: float = 32.0,
                 monitoring_interval: float = 30.0,
                 alert_threshold: float = 0.85):
        """
        Initialize memory monitor

        Args:
            max_memory_gb: Maximum allowed memory usage
            monitoring_interval: Monitoring interval in seconds
            alert_threshold: Memory usage threshold for alerts (0-1)
        """

        self.max_memory_bytes = max_memory_gb * 1024 ** 3
        self.monitoring_interval = monitoring_interval
        self.alert_threshold = alert_threshold

        self.logger = get_logger()
        self.process = psutil.Process()

        # Monitoring data
        self.memory_history: List[MemorySnapshot] = []
        self.max_history_size = 1000

        # Callbacks
        self.cleanup_callbacks: List[Callable] = []
        self.alert_callbacks: List[Callable] = []

        # Monitoring thread
        self.monitoring_thread: Optional[threading.Thread] = None
        self.monitoring_active = False
        self._stop_event = threading.Event()

        # Statistics
        self.stats = {
            'peak_memory_mb': 0.0,
            'alerts_triggered': 0,
            'cleanups_performed': 0,
            'monitoring_start_time': None
        }

    def start_monitoring(self):
        """Start continuous memory monitoring"""

        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.stats['monitoring_start_time'] = time.time()
        self._stop_event.clear()

        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()

        self.logger.info(f"Memory monitoring started (max: {self.max_memory_bytes / 1024 ** 3:.1f}GB, "
                         f"alert: {self.alert_threshold:.0%})")

    def stop_monitoring(self):
        """Stop memory monitoring"""

        if not self.monitoring_active:
            return

        self.monitoring_active = False
        self._stop_event.set()

        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5.0)

        self.logger.info("Memory monitoring stopped")

    def _monitoring_loop(self):
        """Main monitoring loop"""

        while not self._stop_event.wait(self.monitoring_interval):
            try:
                snapshot = self.take_memory_snapshot()
                self.memory_history.append(snapshot)

                # Maintain history size
                if len(self.memory_history) > self.max_history_size:
                    self.memory_history.pop(0)

                # Update peak memory
                current_mb = snapshot.resident_memory_mb
                if current_mb > self.stats['peak_memory_mb']:
                    self.stats['peak_memory_mb'] = current_mb

                # Check thresholds
                self._check_memory_thresholds(snapshot)

            except Exception as e:
                self.logger.error(f"Memory monitoring error: {e}")

    def take_memory_snapshot(self) -> MemorySnapshot:
        """Take current memory usage snapshot"""

        try:
            # Process memory info
            memory_info = self.process.memory_info()
            memory_percent = self.process.memory_percent()

            # System memory info
            virtual_memory = psutil.virtual_memory()

            return MemorySnapshot(
                timestamp=time.time(),
                virtual_memory_mb=memory_info.vms / 1024 ** 2,
                resident_memory_mb=memory_info.rss / 1024 ** 2,
                peak_memory_mb=getattr(memory_info, 'peak_wset', memory_info.rss) / 1024 ** 2,
                available_memory_mb=virtual_memory.available / 1024 ** 2,
                memory_percent=memory_percent,
                swap_memory_mb=getattr(memory_info, 'swap', 0) / 1024 ** 2
            )

        except Exception as e:
            self.logger.error(f"Failed to take memory snapshot: {e}")
            return MemorySnapshot(time.time(), 0, 0, 0, 0, 0)

    def _check_memory_thresholds(self, snapshot: MemorySnapshot):
        """Check memory thresholds and trigger actions"""

        memory_usage_ratio = snapshot.resident_memory_mb * 1024 ** 2 / self.max_memory_bytes

        # Alert threshold
        if memory_usage_ratio > self.alert_threshold:
            self.stats['alerts_triggered'] += 1
            self.logger.warning(f"High memory usage: {snapshot.resident_memory_mb:.1f}MB "
                                f"({memory_usage_ratio:.1%} of limit)")

            # Trigger alert callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(snapshot)
                except Exception as e:
                    self.logger.error(f"Alert callback failed: {e}")

        # Cleanup threshold (higher than alert)
        if memory_usage_ratio > 0.95:
            self.logger.warning("Critical memory usage - triggering cleanup")
            self.trigger_cleanup()

    def register_cleanup_callback(self, callback: Callable):
        """Register cleanup callback function"""
        self.cleanup_callbacks.append(callback)

    def register_alert_callback(self, callback: Callable[[MemorySnapshot], None]):
        """Register alert callback function"""
        self.alert_callbacks.append(callback)

    def trigger_cleanup(self) -> bool:
        """Manually trigger cleanup callbacks"""

        cleanup_success = True
        self.stats['cleanups_performed'] += 1

        self.logger.info("Triggering memory cleanup callbacks")

        for i, callback in enumerate(self.cleanup_callbacks):
            try:
                callback()
                self.logger.debug(f"Cleanup callback {i} completed")
            except Exception as e:
                self.logger.error(f"Cleanup callback {i} failed: {e}")
                cleanup_success = False

        # Force garbage collection
        import gc
        collected = gc.collect()
        self.logger.debug(f"Garbage collection freed {collected} objects")

        return cleanup_success

    def get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB"""
        try:
            return self.process.memory_info().rss / 1024 ** 2
        except:
            return 0.0

    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics"""

        current_snapshot = self.take_memory_snapshot()

        stats = self.stats.copy()
        stats.update({
            'current_memory_mb': current_snapshot.resident_memory_mb,
            'current_memory_percent': current_snapshot.memory_percent,
            'available_memory_mb': current_snapshot.available_memory_mb,
            'monitoring_active': self.monitoring_active,
            'history_size': len(self.memory_history),
            'monitoring_duration': time.time() - self.stats['monitoring_start_time'] if self.stats[
                'monitoring_start_time'] else 0
        })

        # Calculate trends if we have history
        if len(self.memory_history) >= 2:
            recent_memories = [s.resident_memory_mb for s in self.memory_history[-10:]]
            stats['memory_trend_mb_per_min'] = (recent_memories[-1] - recent_memories[0]) / len(recent_memories) * 2

        return stats

    def estimate_memory_leak(self) -> Optional[float]:
        """Estimate memory leak rate in MB per hour"""

        if len(self.memory_history) < 10:
            return None

        # Use linear regression on recent history
        recent_snapshots = self.memory_history[-50:]  # Last 50 snapshots

        times = [s.timestamp for s in recent_snapshots]
        memories = [s.resident_memory_mb for s in recent_snapshots]

        # Simple linear regression
        n = len(times)
        sum_time = sum(times)
        sum_memory = sum(memories)
        sum_time_memory = sum(t * m for t, m in zip(times, memories))
        sum_time_squared = sum(t * t for t in times)

        denominator = n * sum_time_squared - sum_time * sum_time
        if abs(denominator) < 1e-10:
            return None

        slope = (n * sum_time_memory - sum_time * sum_memory) / denominator

        # Convert to MB per hour
        return slope * 3600


class ResourceTracker:
    """
    Comprehensive resource usage tracker

    Extends memory monitoring to include CPU, disk, and network usage
    for complete system resource awareness.
    """

    def __init__(self, monitoring_interval: float = 60.0):
        self.monitoring_interval = monitoring_interval
        self.logger = get_logger()

        # Resource history
        self.resource_history: List[ResourceSnapshot] = []
        self.max_history_size = 500

        # Monitoring control
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Baseline measurements
        self._baseline_disk_io = None
        self._baseline_network_io = None

    def start_tracking(self):
        """Start comprehensive resource tracking"""

        if self.monitoring_active:
            return

        self._establish_baselines()

        self.monitoring_active = True
        self._stop_event.clear()

        self.monitoring_thread = threading.Thread(target=self._tracking_loop, daemon=True)
        self.monitoring_thread.start()

        self.logger.info("Resource tracking started")

    def stop_tracking(self):
        """Stop resource tracking"""

        if not self.monitoring_active:
            return

        self.monitoring_active = False
        self._stop_event.set()

        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)

        self.logger.info("Resource tracking stopped")

    def _establish_baselines(self):
        """Establish baseline measurements for incremental tracking"""

        try:
            self._baseline_disk_io = psutil.disk_io_counters()
            self._baseline_network_io = psutil.net_io_counters()
        except:
            self.logger.warning("Could not establish I/O baselines")

    def _tracking_loop(self):
        """Main resource tracking loop"""

        while not self._stop_event.wait(self.monitoring_interval):
            try:
                snapshot = self.take_resource_snapshot()
                self.resource_history.append(snapshot)

                # Maintain history size
                if len(self.resource_history) > self.max_history_size:
                    self.resource_history.pop(0)

            except Exception as e:
                self.logger.error(f"Resource tracking error: {e}")

    def take_resource_snapshot(self) -> ResourceSnapshot:
        """Take comprehensive resource snapshot"""

        # Memory snapshot
        process = psutil.Process()
        memory_info = process.memory_info()
        virtual_memory = psutil.virtual_memory()

        memory_snapshot = MemorySnapshot(
            timestamp=time.time(),
            virtual_memory_mb=memory_info.vms / 1024 ** 2,
            resident_memory_mb=memory_info.rss / 1024 ** 2,
            peak_memory_mb=getattr(memory_info, 'peak_wset', memory_info.rss) / 1024 ** 2,
            available_memory_mb=virtual_memory.available / 1024 ** 2,
            memory_percent=process.memory_percent()
        )

        # CPU usage
        cpu_percent = process.cpu_percent()

        # Disk I/O (incremental)
        disk_read_mb = disk_write_mb = 0.0
        try:
            current_disk_io = psutil.disk_io_counters()
            if self._baseline_disk_io:
                disk_read_mb = (current_disk_io.read_bytes - self._baseline_disk_io.read_bytes) / 1024 ** 2
                disk_write_mb = (current_disk_io.write_bytes - self._baseline_disk_io.write_bytes) / 1024 ** 2
        except:
            pass

        # Network I/O (incremental)
        net_sent_mb = net_recv_mb = 0.0
        try:
            current_net_io = psutil.net_io_counters()
            if self._baseline_network_io:
                net_sent_mb = (current_net_io.bytes_sent - self._baseline_network_io.bytes_sent) / 1024 ** 2
                net_recv_mb = (current_net_io.bytes_recv - self._baseline_network_io.bytes_recv) / 1024 ** 2
        except:
            pass

        return ResourceSnapshot(
            timestamp=time.time(),
            memory=memory_snapshot,
            cpu_percent=cpu_percent,
            disk_io_read_mb=disk_read_mb,
            disk_io_write_mb=disk_write_mb,
            network_io_sent_mb=net_sent_mb,
            network_io_recv_mb=net_recv_mb
        )

    def get_resource_summary(self) -> Dict[str, Any]:
        """Get comprehensive resource usage summary"""

        if not self.resource_history:
            return {}

        recent_snapshots = self.resource_history[-10:]  # Last 10 snapshots

        # Memory statistics
        memory_usage = [s.memory.resident_memory_mb for s in recent_snapshots]

        # CPU statistics
        cpu_usage = [s.cpu_percent for s in recent_snapshots]

        # I/O statistics
        total_disk_read = self.resource_history[-1].disk_io_read_mb if self.resource_history else 0
        total_disk_write = self.resource_history[-1].disk_io_write_mb if self.resource_history else 0
        total_net_sent = self.resource_history[-1].network_io_sent_mb if self.resource_history else 0
        total_net_recv = self.resource_history[-1].network_io_recv_mb if self.resource_history else 0

        return {
            'memory': {
                'current_mb': memory_usage[-1] if memory_usage else 0,
                'average_mb': sum(memory_usage) / len(memory_usage) if memory_usage else 0,
                'peak_mb': max(memory_usage) if memory_usage else 0,
            },
            'cpu': {
                'current_percent': cpu_usage[-1] if cpu_usage else 0,
                'average_percent': sum(cpu_usage) / len(cpu_usage) if cpu_usage else 0,
                'peak_percent': max(cpu_usage) if cpu_usage else 0,
            },
            'io': {
                'total_disk_read_mb': total_disk_read,
                'total_disk_write_mb': total_disk_write,
                'total_network_sent_mb': total_net_sent,
                'total_network_recv_mb': total_net_recv,
            },
            'tracking_duration': (self.resource_history[-1].timestamp - self.resource_history[0].timestamp) / 60 if len(
                self.resource_history) > 1 else 0
        }
