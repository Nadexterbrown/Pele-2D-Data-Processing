"""
MPI Coordination and Work Distribution

MPI coordination system that handles work distribution,
result collection, and error handling for parallel PELE processing.
"""

import time
import numpy as np
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass
from enum import Enum

try:
    from mpi4py import MPI

    MPI_AVAILABLE = True
except ImportError:
    MPI_AVAILABLE = False
    MPI = None

from ..logging.mpi_logger import get_logger


class WorkDistributionStrategy(Enum):
    """Work distribution strategies"""
    ROUND_ROBIN = "round_robin"
    LOAD_BALANCED = "load_balanced"
    DYNAMIC = "dynamic"
    MASTER_SLAVE = "master_slave"


@dataclass
class WorkItem:
    """Individual work item for processing"""
    id: str
    data: Any
    priority: int = 0
    estimated_time: float = 1.0
    dependencies: List[str] = None

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


@dataclass
class WorkResult:
    """Result from processing a work item"""
    work_id: str
    result: Any
    success: bool
    processing_time: float
    error_message: Optional[str] = None
    rank: int = 0


class MPICoordinator:
    """
    Professional MPI coordinator for parallel processing

    Handles work distribution, progress tracking, and result collection
    with robust error handling and load balancing.
    """

    def __init__(self, strategy: WorkDistributionStrategy = WorkDistributionStrategy.LOAD_BALANCED):
        if not MPI_AVAILABLE:
            raise RuntimeError("MPI not available - cannot create MPICoordinator")

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.strategy = strategy
        self.logger = get_logger()

        # Performance tracking
        self.start_time = None
        self.work_stats = {
            'total_work_items': 0,
            'completed_items': 0,
            'failed_items': 0,
            'total_processing_time': 0.0
        }

    def distribute_work(self, work_items: List[WorkItem],
                        processor_func: Callable[[WorkItem], WorkResult],
                        progress_callback: Optional[Callable] = None) -> List[WorkResult]:
        """
        Distribute work among MPI ranks and collect results

        Args:
            work_items: List of work items to process
            processor_func: Function to process each work item
            progress_callback: Optional progress callback function

        Returns:
            List of work results from all ranks
        """

        self.start_time = time.time()
        self.work_stats['total_work_items'] = len(work_items)

        if self.rank == 0:
            self.logger.info(f"Distributing {len(work_items)} work items across {self.size} ranks")

        # Distribute work based on strategy
        if self.strategy == WorkDistributionStrategy.ROUND_ROBIN:
            results = self._round_robin_processing(work_items, processor_func, progress_callback)
        elif self.strategy == WorkDistributionStrategy.LOAD_BALANCED:
            results = self._load_balanced_processing(work_items, processor_func, progress_callback)
        elif self.strategy == WorkDistributionStrategy.DYNAMIC:
            results = self._dynamic_processing(work_items, processor_func, progress_callback)
        elif self.strategy == WorkDistributionStrategy.MASTER_SLAVE:
            results = self._master_slave_processing(work_items, processor_func, progress_callback)
        else:
            raise ValueError(f"Unknown work distribution strategy: {self.strategy}")

        # Collect final statistics
        self._finalize_processing()

        return results

    def _round_robin_processing(self, work_items: List[WorkItem],
                                processor_func: Callable,
                                progress_callback: Optional[Callable]) -> List[WorkResult]:
        """Simple round-robin work distribution"""

        my_work = [item for i, item in enumerate(work_items) if i % self.size == self.rank]

        self.logger.info(f"Rank {self.rank} processing {len(my_work)} items")

        # Process my work items
        my_results = []
        for work_item in my_work:
            result = self._process_work_item(work_item, processor_func)
            my_results.append(result)

            if progress_callback:
                progress_callback(result)

        # Gather all results
        all_results = self.comm.gather(my_results, root=0)

        # Flatten results on rank 0
        if self.rank == 0:
            flattened_results = []
            for rank_results in all_results:
                flattened_results.extend(rank_results)
            return flattened_results
        else:
            return my_results

    def _load_balanced_processing(self, work_items: List[WorkItem],
                                  processor_func: Callable,
                                  progress_callback: Optional[Callable]) -> List[WorkResult]:
        """Load-balanced work distribution based on estimated times"""

        # Sort work items by estimated time (longest first)
        sorted_items = sorted(work_items, key=lambda x: x.estimated_time, reverse=True)

        # Distribute work to balance load
        rank_loads = [0.0] * self.size
        rank_assignments = [[] for _ in range(self.size)]

        for item in sorted_items:
            # Assign to least loaded rank
            min_rank = np.argmin(rank_loads)
            rank_assignments[min_rank].append(item)
            rank_loads[min_rank] += item.estimated_time

        # Process assigned work
        my_work = rank_assignments[self.rank]
        self.logger.info(f"Rank {self.rank} assigned {len(my_work)} items "
                         f"(estimated time: {rank_loads[self.rank]:.2f}s)")

        my_results = []
        for work_item in my_work:
            result = self._process_work_item(work_item, processor_func)
            my_results.append(result)

            if progress_callback:
                progress_callback(result)

        # Gather results
        all_results = self.comm.gather(my_results, root=0)

        if self.rank == 0:
            flattened_results = []
            for rank_results in all_results:
                flattened_results.extend(rank_results)
            return flattened_results
        else:
            return my_results

    def _dynamic_processing(self, work_items: List[WorkItem],
                            processor_func: Callable,
                            progress_callback: Optional[Callable]) -> List[WorkResult]:
        """Dynamic work distribution - work stealing approach"""

        if self.rank == 0:
            # Master rank manages work queue
            return self._master_work_manager(work_items, processor_func, progress_callback)
        else:
            # Worker ranks request work dynamically
            return self._worker_request_work(processor_func, progress_callback)

    def _master_slave_processing(self, work_items: List[WorkItem],
                                 processor_func: Callable,
                                 progress_callback: Optional[Callable]) -> List[WorkResult]:
        """Master-slave processing pattern"""

        if self.rank == 0:
            return self._master_coordinate_work(work_items, processor_func, progress_callback)
        else:
            return self._slave_process_work(processor_func, progress_callback)

    def _process_work_item(self, work_item: WorkItem,
                           processor_func: Callable) -> WorkResult:
        """Process individual work item with error handling"""

        start_time = time.time()

        try:
            result = processor_func(work_item)
            processing_time = time.time() - start_time

            self.work_stats['completed_items'] += 1
            self.work_stats['total_processing_time'] += processing_time

            return WorkResult(
                work_id=work_item.id,
                result=result,
                success=True,
                processing_time=processing_time,
                rank=self.rank
            )

        except Exception as e:
            processing_time = time.time() - start_time

            self.work_stats['failed_items'] += 1
            self.logger.error(f"Work item {work_item.id} failed: {e}")

            return WorkResult(
                work_id=work_item.id,
                result=None,
                success=False,
                processing_time=processing_time,
                error_message=str(e),
                rank=self.rank
            )

    def _finalize_processing(self):
        """Finalize processing and report statistics"""

        total_time = time.time() - self.start_time if self.start_time else 0

        # Gather statistics from all ranks
        all_stats = self.comm.gather(self.work_stats, root=0)

        if self.rank == 0:
            # Aggregate statistics
            total_completed = sum(stats['completed_items'] for stats in all_stats)
            total_failed = sum(stats['failed_items'] for stats in all_stats)
            total_proc_time = sum(stats['total_processing_time'] for stats in all_stats)

            self.logger.info(
                f"Processing complete: {total_completed} successful, {total_failed} failed "
                f"in {total_time:.2f}s (efficiency: {total_proc_time / total_time / self.size:.2%})"
            )

    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get processing statistics"""
        stats = self.work_stats.copy()
        stats.update({
            'rank': self.rank,
            'size': self.size,
            'strategy': self.strategy.value,
            'elapsed_time': time.time() - self.start_time if self.start_time else 0
        })
        return stats


class MPIWorkDistributor:
    """
    Simplified work distributor for common use cases

    Provides easy-to-use interface for common parallel processing patterns
    without the complexity of the full MPICoordinator.
    """

    def __init__(self):
        if not MPI_AVAILABLE:
            self.comm = None
            self.rank = 0
            self.size = 1
        else:
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()

        self.logger = get_logger()

    def parallel_map(self, func: Callable, items: List[Any]) -> List[Any]:
        """
        Simple parallel map function

        Args:
            func: Function to apply to each item
            items: List of items to process

        Returns:
            List of results
        """

        # Distribute items round-robin
        my_items = [item for i, item in enumerate(items) if i % self.size == self.rank]

        # Process my items
        my_results = [func(item) for item in my_items]

        if not MPI_AVAILABLE:
            return my_results

        # Gather results
        all_results = self.comm.gather(my_results, root=0)

        if self.rank == 0:
            # Reconstruct original order
            final_results = [None] * len(items)
            for rank, rank_results in enumerate(all_results):
                rank_indices = [i for i in range(rank, len(items), self.size)]
                for idx, result in zip(rank_indices, rank_results):
                    final_results[idx] = result
            return final_results
        else:
            return []

    def parallel_reduce(self, func: Callable, items: List[Any],
                        reduce_func: Callable) -> Any:
        """
        Parallel map-reduce operation

        Args:
            func: Map function
            items: Items to process
            reduce_func: Reduce function

        Returns:
            Reduced result
        """

        # Parallel map phase
        mapped_results = self.parallel_map(func, items)

        if self.rank == 0:
            # Reduce phase (only on rank 0)
            if mapped_results:
                result = mapped_results[0]
                for item in mapped_results[1:]:
                    result = reduce_func(result, item)
                return result
            else:
                return None
        else:
            return None