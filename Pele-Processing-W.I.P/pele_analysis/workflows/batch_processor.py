"""Batch processing workflow"""

from typing import List, Dict, Any, Optional
import time
from pathlib import Path
from mpi4py import MPI

from ..core.data_structures import ProcessedData
from ..core.io.pele_loader import PeleDataLoader
from ..core.io.file_manager import FileManager
from ..core.io.output_writer import OutputWriter
from ..physics.wave_tracking import WaveTracker
from ..config.processing_config import ProcessingConfiguration
from ..infrastructure.logging.mpi_logger import get_logger

logger = get_logger(__name__)


class BatchProcessor:
    """Professional batch processing workflow"""

    def __init__(self, config: ProcessingConfiguration):
        self.config = config

        # Initialize MPI
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        # Initialize components
        self.data_loader = PeleDataLoader()
        self.wave_tracker = WaveTracker(config.flame_temp_threshold)
        self.file_manager = FileManager()
        self.output_writer = OutputWriter(config.output_directory)

    def process_directory(self, data_directory: str) -> Dict[str, Any]:
        """Process entire directory of PELE data"""

        start_time = time.time()

        if self.rank == 0:
            logger.log(f"Starting batch processing: {data_directory}")

        # Load and sort files
        plt_files = self.file_manager.load_pele_directories(data_directory)
        sorted_files = self.file_manager.sort_plt_files(plt_files)

        if self.rank == 0:
            logger.log(f"Found {len(sorted_files)} files to process")

        # Process files in parallel
        results = self._process_files_parallel(sorted_files)

        # Collect results
        all_results = self.comm.gather(results, root=0)

        if self.rank == 0:
            # Merge and save results
            final_results = self._merge_results(all_results)
            self._save_results(final_results)

            elapsed = time.time() - start_time
            logger.log(f"Batch processing completed in {elapsed:.2f} seconds")

            return final_results

        return {}

    def _process_files_parallel(self, file_list: List[str]) -> Dict[str, Any]:
        """Process files using MPI parallelization"""

        results = {}
        my_files = [f for i, f in enumerate(file_list) if i % self.size == self.rank]

        logger.log(f"Rank {self.rank} processing {len(my_files)} files")

        for file_path in my_files:
            try:
                result = self._process_single_file(file_path)
                if result:
                    results[Path(file_path).name] = result
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")

        return results