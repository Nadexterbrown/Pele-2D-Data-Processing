"""
MPI-Aware Logging System

Professional logging system that handles MPI coordination with proper
file locking, rank identification, and performance optimization.
"""

import os
import logging
import threading
from datetime import datetime
from pathlib import Path
from io import StringIO
from typing import Optional, Dict, Any
from contextlib import contextmanager
import fcntl
import time

try:
    from mpi4py import MPI

    MPI_AVAILABLE = True
except ImportError:
    MPI_AVAILABLE = False
    MPI = None


class MPILogger:
    """
    Professional MPI-aware logging system

    Provides thread-safe, rank-aware logging with proper file coordination
    and performance optimization for high-performance computing environments.
    """

    def __init__(self, log_file: str = "pele_processing.log",
                 overwrite: bool = False, buffer_size: int = 1000,
                 flush_interval: float = 10.0):
        """
        Initialize MPI logger

        Args:
            log_file: Path to log file
            overwrite: Whether to overwrite existing log file
            buffer_size: Size of log buffer before auto-flush
            flush_interval: Time interval for auto-flush (seconds)
        """

        # MPI setup
        if MPI_AVAILABLE:
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()
        else:
            self.comm = None
            self.rank = 0
            self.size = 1

        # File setup
        self.log_file_path = Path(log_file)
        self.log_file_path.parent.mkdir(parents=True, exist_ok=True)

        # Buffer setup
        self.buffer = StringIO()
        self.buffer_size = buffer_size
        self.buffer_count = 0
        self.flush_interval = flush_interval
        self.last_flush_time = time.time()

        # Thread safety
        self._lock = threading.Lock()

        # Statistics
        self.stats = {
            'messages_logged': 0,
            'bytes_written': 0,
            'flush_count': 0,
            'errors': 0
        }

        # Initialize log file
        self._initialize_log_file(overwrite)

        # Setup Python logging integration
        self._setup_python_logging()

    def _initialize_log_file(self, overwrite: bool):
        """Initialize log file with header"""

        # Only rank 0 initializes the file
        if self.rank == 0:
            if overwrite and self.log_file_path.exists():
                self.log_file_path.unlink()

            # Write header
            header_lines = [
                f"===== PELE Analysis Processing Log =====",
                f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"MPI Size: {self.size} ranks",
                f"Log File: {self.log_file_path}",
                "=" * 50
            ]

            try:
                with self._acquire_file_lock('a') as f:
                    for line in header_lines:
                        f.write(f"{line}\n")
                    f.flush()
            except Exception as e:
                print(f"Failed to initialize log file: {e}")

        # Synchronize ranks
        if MPI_AVAILABLE:
            self.comm.Barrier()

        # Each rank logs initialization
        self.log(f"Rank {self.rank} initialized successfully", level="INFO")
        self.flush()

    def log(self, message: str, level: str = "INFO", category: str = None):
        """
        Log message with rank and timestamp

        Args:
            message: Message to log
            level: Log level (INFO, WARNING, ERROR, DEBUG)
            category: Optional category for message
        """

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        # Format message
        category_str = f"[{category}] " if category else ""
        formatted_msg = f"{timestamp} [Rank {self.rank:2d}] {level:7s} - {category_str}{message}"

        with self._lock:
            self.buffer.write(f"{formatted_msg}\n")
            self.buffer_count += 1
            self.stats['messages_logged'] += 1

            # Auto-flush based on buffer size or time
            current_time = time.time()
            if (self.buffer_count >= self.buffer_size or
                    current_time - self.last_flush_time >= self.flush_interval):
                self._flush_buffer()

    def info(self, message: str, category: str = None):
        """Log info message"""
        self.log(message, "INFO", category)

    def warning(self, message: str, category: str = None):
        """Log warning message"""
        self.log(message, "WARNING", category)

    def error(self, message: str, category: str = None):
        """Log error message"""
        self.log(message, "ERROR", category)
        self.stats['errors'] += 1

    def debug(self, message: str, category: str = None):
        """Log debug message"""
        self.log(message, "DEBUG", category)

    def flush(self):
        """Force flush buffer to file"""
        with self._lock:
            self._flush_buffer()

    def _flush_buffer(self):
        """Internal buffer flush implementation"""

        if self.buffer_count == 0:
            return

        buffer_content = self.buffer.getvalue()
        if not buffer_content:
            return

        try:
            with self._acquire_file_lock('a') as f:
                f.write(buffer_content)
                f.flush()
                os.fsync(f.fileno())  # Force write to disk

            # Update statistics
            self.stats['bytes_written'] += len(buffer_content)
            self.stats['flush_count'] += 1

        except Exception as e:
            print(f"Rank {self.rank}: Failed to flush log buffer: {e}")
            self.stats['errors'] += 1
        finally:
            # Clear buffer
            self.buffer.truncate(0)
            self.buffer.seek(0)
            self.buffer_count = 0
            self.last_flush_time = time.time()

    @contextmanager
    def _acquire_file_lock(self, mode: str):
        """Context manager for file locking"""

        f = None
        try:
            f = open(self.log_file_path, mode, buffering=1)

            # File locking for coordination between ranks
            if hasattr(fcntl, 'flock'):
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)

            yield f

        except Exception as e:
            print(f"Rank {self.rank}: File lock acquisition failed: {e}")
            raise
        finally:
            if f:
                try:
                    if hasattr(fcntl, 'flock'):
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                    f.close()
                except:
                    pass

    def _setup_python_logging(self):
        """Setup Python logging integration"""

        # Create custom handler
        handler = MPILogHandler(self)
        handler.setLevel(logging.DEBUG)

        # Create formatter
        formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        # Setup root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.INFO)

        # Setup pele-specific loggers
        for name in ['PeleAnalysis', 'WaveTracker', 'FlameAnalyzer']:
            logger = logging.getLogger(name)
            logger.setLevel(logging.INFO)

    def get_statistics(self) -> Dict[str, Any]:
        """Get logging statistics"""
        stats = self.stats.copy()
        stats.update({
            'buffer_size': self.buffer_count,
            'log_file_size': self.log_file_path.stat().st_size if self.log_file_path.exists() else 0,
            'rank': self.rank,
            'total_ranks': self.size
        })
        return stats

    def finalize(self):
        """Finalize logging and cleanup"""
        self.flush()

        if self.rank == 0:
            footer_lines = [
                "=" * 50,
                f"Processing completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"Total messages logged: {sum(self.stats['messages_logged'] for _ in range(self.size))} (estimated)",
                "===== End of Log ====="
            ]

            try:
                with self._acquire_file_lock('a') as f:
                    for line in footer_lines:
                        f.write(f"{line}\n")
            except Exception as e:
                print(f"Failed to write log footer: {e}")


class MPILogHandler(logging.Handler):
    """Custom logging handler for MPI logger integration"""

    def __init__(self, mpi_logger: MPILogger):
        super().__init__()
        self.mpi_logger = mpi_logger

    def emit(self, record):
        try:
            level = record.levelname
            message = self.format(record)
            self.mpi_logger.log(message, level)
        except:
            self.handleError(record)


# Global logger instance
_global_logger: Optional[MPILogger] = None


def setup_logging(log_file: str = "pele_processing.log",
                  overwrite: bool = False,
                  verbose: bool = False) -> MPILogger:
    """
    Setup global MPI logging

    Args:
        log_file: Path to log file
        overwrite: Whether to overwrite existing log
        verbose: Enable verbose logging

    Returns:
        MPILogger instance
    """
    global _global_logger

    if _global_logger is None:
        _global_logger = MPILogger(log_file, overwrite)

    if verbose:
        # Set debug level for verbose output
        logging.getLogger().setLevel(logging.DEBUG)

    return _global_logger


def get_logger(name: str = None) -> MPILogger:
    """Get global logger instance"""
    if _global_logger is None:
        setup_logging()

    return _global_logger