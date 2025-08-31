"""
Cache Manager with YT Parallelization Support

This module provides caching that works with both manual MPI and yt's built-in
parallelization methods, with configuration options for different approaches.
"""

import os
import pickle
import hashlib
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import yt
from mpi4py import MPI


# ===================================================================
# PARALLELIZATION STRATEGY CONFIGURATION
# ===================================================================

class ParallelStrategy(Enum):
    """Different parallelization approaches"""
    MANUAL_MPI = "manual_mpi"  # Manual MPI coordination
    YT_PARALLEL = "yt_parallel"  # Use yt.parallel_objects()
    HYBRID = "hybrid"  # Combine both approaches
    SINGLE_PROCESS = "single_process"  # No parallelization


@dataclass
class CacheConfig:
    """Configuration for cache behavior with different parallel strategies"""

    # Basic cache settings
    cache_dir: str = "./cache"
    max_size_gb: float = 10.0

    # Parallelization strategy
    parallel_strategy: ParallelStrategy = ParallelStrategy.MANUAL_MPI

    # Strategy-specific settings
    use_rank_specific_cache: bool = False  # Separate cache per MPI rank
    use_shared_cache: bool = True  # Shared cache across ranks
    cache_coordination_method: str = "file_lock"  # "file_lock", "mpi_barrier", "none"

    # YT-specific settings
    yt_enable_parallelism: bool = True  # Call yt.enable_parallelism()
    yt_storage_sharing: bool = True  # Share yt storage objects

    # Performance tuning
    cache_write_delay: float = 0.1  # Delay to avoid write conflicts
    max_cache_attempts: int = 3  # Max attempts for cache operations

    def validate(self) -> bool:
        """Validate configuration"""
        if self.max_size_gb <= 0:
            raise ValueError("Cache size must be positive")

        if self.parallel_strategy == ParallelStrategy.YT_PARALLEL and not self.yt_enable_parallelism:
            raise ValueError("YT parallel strategy requires yt_enable_parallelism=True")

        return True


# ===================================================================
# BASE CACHE MANAGER
# ===================================================================

class BaseCacheManager:
    """Base cache manager with common functionality"""

    def __init__(self, config: CacheConfig):
        self.config = config
        self.config.validate()

        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.metadata_file = self.cache_dir / "metadata.pkl"
        self.lock_file = self.cache_dir / ".cache_lock"

        # MPI info (available regardless of strategy)
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        # Load metadata
        self.metadata = self._load_metadata()

    def get_cache_key(self, file_path: Union[str, Path],
                      extraction_params: Any) -> str:
        """Generate unique cache key"""
        file_path = Path(file_path)

        # Include file modification time and extraction parameters
        try:
            mtime = file_path.stat().st_mtime
        except (OSError, FileNotFoundError):
            mtime = 0

        key_string = f"{file_path.name}:{mtime}:{hash(str(extraction_params))}"
        return hashlib.md5(key_string.encode()).hexdigest()

    def _load_metadata(self) -> Dict[str, Any]:
        """Load cache metadata with proper locking"""
        if not self.metadata_file.exists():
            return {}

        try:
            with self._acquire_file_lock(self.metadata_file, 'rb') as f:
                return pickle.load(f)
        except (EOFError, pickle.UnpicklingError):
            return {}

    def _save_metadata(self):
        """Save metadata with proper locking"""
        try:
            with self._acquire_file_lock(self.metadata_file, 'wb') as f:
                pickle.dump(self.metadata, f)
        except Exception as e:
            print(f"Warning: Could not save cache metadata: {e}")

    def _acquire_file_lock(self, file_path: Path, mode: str):
        """Context manager for file locking"""
        from contextlib import contextmanager

        try:
            import fcntl  # Unix
            _HAS_FCNTL = True
        except ImportError:
            fcntl = None
            _HAS_FCNTL = False

        try:
            import msvcrt  # Windows
            _HAS_MSVCRT = True
        except ImportError:
            msvcrt = None
            _HAS_MSVCRT = False

        @contextmanager
        def locked_file():
            """
            Open a file and lock it (advisory) while the context is held.

            - Unix: fcntl.flock(..., LOCK_EX)
            - Windows: msvcrt.locking(..., LK_LOCK) on the first byte
            - If `self.config.cache_coordination_method != "file_lock"`, no lock is taken.

            NOTE: Windows byte-range lock needs a nonzero region length; we lock 1 byte at offset 0.
            """
            # If you're calling this as a method and have self.config, keep this guard;
            # otherwise, drop it or replace as needed.
            try:
                use_locking = (self.config.cache_coordination_method == "file_lock")  # noqa: F821
            except NameError:
                use_locking = True  # default to locking if no self/config in scope

            # Ensure binary+read/write by default (Windows text mode can be weird with offsets).
            f = open(file_path, mode)

            locked = False
            try:
                if use_locking:
                    if _HAS_FCNTL:
                        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                        locked = True
                    elif _HAS_MSVCRT:
                        # Lock the first byte. Seek to 0 to define the region start.
                        f.seek(0)
                        msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
                        locked = True
                    else:
                        # No locking available on this platform — you're effectively running unlocked.
                        pass

                yield f

            finally:
                if use_locking and locked:
                    try:
                        if _HAS_FCNTL:
                            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                        elif _HAS_MSVCRT:
                            f.seek(0)
                            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
                    except Exception:
                        # Don’t let unlock errors mask real exceptions from the with-block
                        pass
                f.close()

        return locked_file()


# ===================================================================
# YT PARALLEL CACHE MANAGER
# ===================================================================

class YTParallelCacheManager(BaseCacheManager):
    """Cache manager optimized for yt parallelization"""

    def __init__(self, config: CacheConfig):
        super().__init__(config)

        # Setup YT parallelization if requested
        if self.config.yt_enable_parallelism:
            yt.enable_parallelism()

        # Create rank-specific cache directory if needed
        if self.config.use_rank_specific_cache:
            self.rank_cache_dir = self.cache_dir / f"rank_{self.rank}"
            self.rank_cache_dir.mkdir(exist_ok=True)
        else:
            self.rank_cache_dir = self.cache_dir

    def get_cached_data(self, file_path: Union[str, Path],
                        extraction_params: Any) -> Optional[Any]:
        """Get cached data with yt parallel coordination"""

        cache_key = self.get_cache_key(file_path, extraction_params)

        # Try rank-specific cache first
        if self.config.use_rank_specific_cache:
            result = self._get_from_rank_cache(cache_key)
            if result is not None:
                return result

        # Try shared cache
        if self.config.use_shared_cache:
            result = self._get_from_shared_cache(cache_key)
            if result is not None:
                return result

        return None

    def cache_data(self, file_path: Union[str, Path],
                   extraction_params: Any, data: Any):
        """Cache data with yt parallel coordination"""

        cache_key = self.get_cache_key(file_path, extraction_params)

        # Cache to rank-specific location if configured
        if self.config.use_rank_specific_cache:
            self._cache_to_rank_cache(cache_key, data)

        # Cache to shared location if configured
        if self.config.use_shared_cache:
            # Use coordination to avoid conflicts
            if self.config.cache_coordination_method == "mpi_barrier":
                self._cache_to_shared_cache_with_barrier(cache_key, data)
            else:
                self._cache_to_shared_cache(cache_key, data)

    def _get_from_rank_cache(self, cache_key: str) -> Optional[Any]:
        """Get data from rank-specific cache"""
        cache_file = self.rank_cache_dir / f"{cache_key}.pkl"

        if not cache_file.exists():
            return None

        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception:
            return None

    def _get_from_shared_cache(self, cache_key: str) -> Optional[Any]:
        """Get data from shared cache"""
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        if not cache_file.exists():
            return None

        try:
            with self._acquire_file_lock(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception:
            return None

    def _cache_to_rank_cache(self, cache_key: str, data: Any):
        """Cache to rank-specific location"""
        cache_file = self.rank_cache_dir / f"{cache_key}.pkl"

        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"Rank {self.rank}: Failed to cache to rank cache: {e}")

    def _cache_to_shared_cache(self, cache_key: str, data: Any):
        """Cache to shared location with file locking"""
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        # Add small delay to reduce contention
        time.sleep(self.config.cache_write_delay * self.rank)

        for attempt in range(self.config.max_cache_attempts):
            try:
                with self._acquire_file_lock(cache_file, 'wb') as f:
                    pickle.dump(data, f)

                # Update metadata
                file_size = cache_file.stat().st_size
                self.metadata[cache_key] = {
                    'file': str(cache_file),
                    'size': file_size,
                    'created': time.time(),
                    'rank': self.rank
                }
                self._save_metadata()
                break

            except Exception as e:
                if attempt == self.config.max_cache_attempts - 1:
                    print(f"Rank {self.rank}: Failed to cache after {self.config.max_cache_attempts} attempts: {e}")
                else:
                    time.sleep(0.1 * (attempt + 1))

    def _cache_to_shared_cache_with_barrier(self, cache_key: str, data: Any):
        """Cache to shared location using MPI barriers for coordination"""
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        # Only rank 0 writes to shared cache
        if self.rank == 0:
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(data, f)

                # Update metadata
                file_size = cache_file.stat().st_size
                self.metadata[cache_key] = {
                    'file': str(cache_file),
                    'size': file_size,
                    'created': time.time(),
                    'rank': 0
                }
                self._save_metadata()
            except Exception as e:
                print(f"Rank 0: Failed to cache: {e}")

        # All ranks wait for rank 0 to complete
        self.comm.Barrier()


# ===================================================================
# HYBRID CACHE MANAGER
# ===================================================================

class HybridCacheManager(BaseCacheManager):
    """Cache manager that combines manual MPI with yt parallelization"""

    def __init__(self, config: CacheConfig):
        super().__init__(config)

        # Enable yt parallelization
        if self.config.yt_enable_parallelism:
            yt.enable_parallelism()

        # Create hierarchical cache structure
        self.shared_cache_dir = self.cache_dir / "shared"
        self.rank_cache_dir = self.cache_dir / f"rank_{self.rank}"
        self.temp_cache_dir = self.cache_dir / "temp"

        for cache_dir in [self.shared_cache_dir, self.rank_cache_dir, self.temp_cache_dir]:
            cache_dir.mkdir(exist_ok=True)

    def get_cached_data(self, file_path: Union[str, Path],
                        extraction_params: Any) -> Optional[Any]:
        """Get cached data using hybrid approach"""

        cache_key = self.get_cache_key(file_path, extraction_params)

        # Priority order: rank cache -> shared cache -> temp cache
        for cache_dir in [self.rank_cache_dir, self.shared_cache_dir, self.temp_cache_dir]:
            cache_file = cache_dir / f"{cache_key}.pkl"
            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        return pickle.load(f)
                except Exception:
                    continue

        return None

    def cache_data(self, file_path: Union[str, Path],
                   extraction_params: Any, data: Any,
                   cache_level: str = "auto"):
        """Cache data with level selection"""

        cache_key = self.get_cache_key(file_path, extraction_params)

        if cache_level == "auto":
            # Determine cache level based on data characteristics
            cache_level = self._determine_cache_level(data, extraction_params)

        if cache_level == "rank":
            target_dir = self.rank_cache_dir
        elif cache_level == "shared":
            target_dir = self.shared_cache_dir
        else:  # temp
            target_dir = self.temp_cache_dir

        cache_file = target_dir / f"{cache_key}.pkl"

        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"Rank {self.rank}: Failed to cache to {cache_level}: {e}")

    def _determine_cache_level(self, data: Any, extraction_params: Any) -> str:
        """Determine appropriate cache level for data"""

        # Simple heuristics - you can customize these
        try:
            data_size = len(pickle.dumps(data))

            # Large data or rank-specific computations -> rank cache
            if data_size > 10_000_000 or hasattr(extraction_params, 'rank_specific'):
                return "rank"

            # Medium data that's reusable -> shared cache
            elif data_size > 100_000:
                return "shared"

            # Small, temporary data -> temp cache
            else:
                return "temp"
        except:
            return "temp"


# ===================================================================
# CACHE MANAGER FACTORY
# ===================================================================

def create_cache_manager(config: CacheConfig) -> BaseCacheManager:
    """Factory function to create appropriate cache manager"""

    if config.parallel_strategy == ParallelStrategy.SINGLE_PROCESS:
        return BaseCacheManager(config)

    elif config.parallel_strategy == ParallelStrategy.MANUAL_MPI:
        return BaseCacheManager(config)  # Use base class for manual MPI

    elif config.parallel_strategy == ParallelStrategy.YT_PARALLEL:
        return YTParallelCacheManager(config)

    elif config.parallel_strategy == ParallelStrategy.HYBRID:
        return HybridCacheManager(config)

    else:
        raise ValueError(f"Unknown parallel strategy: {config.parallel_strategy}")


# ===================================================================
# INTEGRATION WITH PELE DATA LOADER
# ===================================================================

class YTParallelDataLoader:
    """PELE data loader with yt parallelization support"""

    def __init__(self, cache_config: CacheConfig):
        self.cache_manager = create_cache_manager(cache_config)
        self.config = cache_config

        # Setup yt parallelization
        if cache_config.yt_enable_parallelism:
            yt.enable_parallelism()

    def extract_data_parallel(self, file_list: list, extraction_params: Any,
                              processing_function) -> dict:
        """Extract data from multiple files using yt parallelization"""

        if self.config.parallel_strategy == ParallelStrategy.YT_PARALLEL:
            return self._extract_with_yt_parallel(file_list, extraction_params, processing_function)
        else:
            return self._extract_with_manual_mpi(file_list, extraction_params, processing_function)

    def _extract_with_yt_parallel(self, file_list: list, extraction_params: Any,
                                  processing_function) -> dict:
        """Use yt.parallel_objects for processing"""

        results = {}

        # yt automatically distributes files among ranks
        for sto, file_path in yt.parallel_objects(file_list, -1, storage=results):

            # Check cache first
            cached_data = self.cache_manager.get_cached_data(file_path, extraction_params)

            if cached_data is not None:
                sto.result_id = os.path.basename(file_path)
                sto.result = cached_data
            else:
                # Process file
                try:
                    processed_data = processing_function(file_path, extraction_params)

                    # Cache result
                    self.cache_manager.cache_data(file_path, extraction_params, processed_data)

                    sto.result_id = os.path.basename(file_path)
                    sto.result = processed_data

                except Exception as e:
                    print(f"Processing failed for {file_path}: {e}")
                    continue

        return results

    def _extract_with_manual_mpi(self, file_list: list, extraction_params: Any,
                                 processing_function) -> dict:
        """Use manual MPI distribution"""

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        # Distribute files manually
        my_files = [f for i, f in enumerate(file_list) if i % size == rank]

        results = {}
        for file_path in my_files:
            # Check cache
            cached_data = self.cache_manager.get_cached_data(file_path, extraction_params)

            if cached_data is not None:
                results[os.path.basename(file_path)] = cached_data
            else:
                try:
                    processed_data = processing_function(file_path, extraction_params)
                    self.cache_manager.cache_data(file_path, extraction_params, processed_data)
                    results[os.path.basename(file_path)] = processed_data
                except Exception as e:
                    print(f"Rank {rank}: Processing failed for {file_path}: {e}")

        # Gather results
        all_results = comm.gather(results, root=0)

        if rank == 0:
            final_results = {}
            for rank_results in all_results:
                final_results.update(rank_results)
            return final_results

        return {}


# ===================================================================
# USAGE EXAMPLES
# ===================================================================

def example_base_mpi_usage():
    """Example of using yt parallelization with caching"""

    # Configuration for yt parallelization
    config = CacheConfig(
        parallel_strategy=ParallelStrategy.MANUAL_MPI,
        use_shared_cache=True,
        cache_coordination_method="file_lock",
        max_size_gb=20.0
    )

    # Create cache manager
    cache_manager = create_cache_manager(config)

    return cache_manager


def example_yt_parallel_usage():
    """Example of using yt parallelization with caching"""

    # Configuration for yt parallelization
    config = CacheConfig(
        parallel_strategy=ParallelStrategy.YT_PARALLEL,
        yt_enable_parallelism=True,
        use_shared_cache=True,
        cache_coordination_method="file_lock",
        max_size_gb=20.0
    )

    # Create loader
    loader = YTParallelDataLoader(config)

    # Define processing function
    def process_file(file_path, params):
        # Your processing logic here
        from pele_loader import extract_pele_data
        return extract_pele_data(file_path, params.location)

    # Process files
    file_list = ["../../../../2D-Test-Data/plt73000", "../../../../2D-Test-Data/plt332320"]  # Your plt files
    extraction_params = type('Params', (), {'location': 0.0445 / 100})()

    results = loader.extract_data_parallel(file_list, extraction_params, process_file)

    return results


def example_hybrid_usage():
    """Example of hybrid approach"""

    # Configuration for hybrid approach
    config = CacheConfig(
        parallel_strategy=ParallelStrategy.HYBRID,
        yt_enable_parallelism=True,
        use_shared_cache=True,
        use_rank_specific_cache=True,
        cache_coordination_method="mpi_barrier"
    )

    # Create cache manager
    cache_manager = create_cache_manager(config)

    # Use with your existing processing logic
    # The cache manager handles coordination automatically

    return cache_manager


if __name__ == "__main__":
    # Test the different approaches
    print("Testing Base MPI Parallel approach...")
    cache_mpi = example_base_mpi_usage()

    print("Testing YT Parallel approach...")
    results_yt = example_yt_parallel_usage()

    print("Testing Hybrid approach...")
    cache_hybrid = example_hybrid_usage()