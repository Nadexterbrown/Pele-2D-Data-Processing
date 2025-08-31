"""
File and Directory Management for PELE Data

Handles file discovery, sorting, path management, and domain parameter extraction.
Replaces the file management functions from general.py with proper error handling.
"""

import os
import re
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass
import numpy as np
import yt


@dataclass
class DirectoryInfo:
    """Information about a PELE data directory"""
    path: Path
    plt_files: List[Path]
    total_size_mb: float
    date_range: Tuple[Optional[float], Optional[float]]  # (start_time, end_time)
    grid_info: Optional[Dict] = None


class FileManager:
    """File and directory management for PELE data"""

    def __init__(self, logger=None):
        self.logger = logger or self._create_default_logger()

    def ensure_long_path_prefix(self, path: Union[str, Path]) -> str:
        """
        Handle Windows long path issues

        Args:
            path: File or directory path

        Returns:
            Path with Windows long path prefix if needed
        """
        if os.name != 'nt':  # Not Windows
            return str(path)

        path = Path(path).resolve()
        path_str = str(path)

        if path_str.startswith(r"\\"):
            return r"\\?\UNC" + path_str[1:]
        return r"\\?\\" + path_str

    def discover_pele_directories(self, parent_dir: Union[str, Path]) -> List[Path]:
        """
        Discover all PELE plt directories in a parent directory

        Args:
            parent_dir: Parent directory containing plt directories

        Returns:
            List of plt directory paths

        Raises:
            FileNotFoundError: If parent directory doesn't exist
            ValueError: If no plt directories found
        """
        parent_path = Path(parent_dir)

        if not parent_path.exists():
            raise FileNotFoundError(f"Directory does not exist: {parent_dir}")

        if not parent_path.is_dir():
            raise ValueError(f"Path is not a directory: {parent_dir}")

        # Find all plt directories
        plt_dirs = []
        try:
            for item in parent_path.iterdir():
                if item.is_dir() and self._is_plt_directory(item.name):
                    plt_dirs.append(item)
        except PermissionError as e:
            raise PermissionError(f"Cannot access directory {parent_dir}: {e}")

        if not plt_dirs:
            raise ValueError(f"No plt directories found in {parent_dir}")

        self.logger.info(f"Discovered {len(plt_dirs)} plt directories in {parent_path.name}")
        return plt_dirs

    def sort_plt_files(self, file_list: List[Union[str, Path]]) -> List[Path]:
        """
        Sort plt files by numeric sequence

        Args:
            file_list: List of plt file/directory paths

        Returns:
            Sorted list of Path objects
        """
        if not file_list:
            return []

        # Convert to Path objects and extract sort keys
        path_objects = [Path(f) for f in file_list]

        def get_sort_key(file_path: Path) -> int:
            """Extract numeric part for sorting"""
            name = file_path.name
            match = re.search(r'plt(\d+)', name)
            if match:
                return int(match.group(1))
            else:
                self.logger.warning(f"Could not extract plt number from {name}")
                return float('inf')  # Put unparseable files at end

        sorted_paths = sorted(path_objects, key=get_sort_key)

        self.logger.debug(f"Sorted {len(sorted_paths)} plt files")
        return sorted_paths

    def get_directory_info(self, directory: Union[str, Path]) -> DirectoryInfo:
        """
        Get comprehensive information about a PELE data directory

        Args:
            directory: Path to directory containing plt files

        Returns:
            DirectoryInfo object with metadata
        """
        dir_path = Path(directory)

        # Discover and sort plt files
        plt_files = self.discover_pele_directories(dir_path)
        sorted_files = self.sort_plt_files(plt_files)

        # Calculate total size
        total_size = 0
        for plt_file in sorted_files:
            try:
                if plt_file.is_file():
                    total_size += plt_file.stat().st_size
                elif plt_file.is_dir():
                    total_size += sum(f.stat().st_size for f in plt_file.rglob('*') if f.is_file())
            except (OSError, PermissionError):
                continue

        total_size_mb = total_size / (1024 * 1024)

        # Extract time range (would need to load files for actual times)
        date_range = self._estimate_time_range(sorted_files)

        return DirectoryInfo(
            path=dir_path,
            plt_files=sorted_files,
            total_size_mb=total_size_mb,
            date_range=date_range
        )

    def extract_domain_parameters(self, plt_file: Union[str, Path],
                                  y_location: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Extract domain parameters from a PELE plt file

        Args:
            plt_file: Path to plt file
            y_location: Desired y-location for extraction (meters), None for center

        Returns:
            Tuple of (slice_indices, slice_coordinates, grid_dict)
        """
        plt_path = Path(plt_file)

        if not plt_path.exists():
            raise FileNotFoundError(f"PLT file not found: {plt_file}")

        try:
            with yt.load(str(plt_path)) as ds:
                max_level = ds.index.max_level

                # Get domain bounds (convert from cm to meters)
                x_min = ds.domain_left_edge[0].to_value() / 100
                x_max = ds.domain_right_edge[0].to_value() / 100
                y_min = ds.domain_left_edge[1].to_value() / 100
                y_max = ds.domain_right_edge[1].to_value() / 100

                # Create coordinate arrays
                dims = ds.domain_dimensions
                nx = dims[0] * (2 ** max_level)
                ny = dims[1] * (2 ** max_level)

                x_coords = np.linspace(x_min, x_max, nx)
                y_coords = np.linspace(y_min, y_max, ny)

                grid_dict = {
                    'x': x_coords,
                    'y': y_coords,
                    'nx': nx,
                    'ny': ny,
                    'max_level': max_level
                }

                # Determine y-slice location and index
                if y_location is None:
                    # Use center
                    y_slice_idx = ny // 2
                    y_slice_loc = y_coords[y_slice_idx]
                else:
                    # Find closest index to requested location
                    y_slice_idx = np.argmin(np.abs(y_coords - y_location))
                    y_slice_loc = y_coords[y_slice_idx]

                # Create slice indices and physical coordinates
                slice_indices = np.array([[0, y_slice_idx], [nx - 1, y_slice_idx]])
                slice_coordinates = np.array([
                    [x_min, y_slice_loc],
                    [x_max, y_slice_loc]
                ])

                self.logger.info(f"Extracted domain parameters: y-slice at {y_slice_loc:.6f}m")

                return slice_indices, slice_coordinates, grid_dict

        except Exception as e:
            raise RuntimeError(f"Failed to extract domain parameters from {plt_file}: {e}")

    def validate_plt_file(self, plt_file: Union[str, Path]) -> bool:
        """
        Validate that a file is a proper PELE plt file

        Args:
            plt_file: Path to plt file

        Returns:
            True if valid, False otherwise
        """
        plt_path = Path(plt_file)

        if not plt_path.exists():
            return False

        # Check if it's a plt directory or file
        if not self._is_plt_directory(plt_path.name):
            return False

        try:
            # Try to load with yt
            with yt.load(str(plt_path)) as ds:
                # Basic validation - has required fields
                return hasattr(ds, 'domain_dimensions') and hasattr(ds, 'current_time')
        except:
            return False

    def cleanup_cache_files(self, directory: Union[str, Path],
                            older_than_hours: int = 24) -> int:
        """
        Clean up old cache files in directory

        Args:
            directory: Directory to clean
            older_than_hours: Remove files older than this many hours

        Returns:
            Number of files removed
        """
        dir_path = Path(directory)

        if not dir_path.exists():
            return 0

        import time
        current_time = time.time()
        cutoff_time = current_time - (older_than_hours * 3600)

        removed_count = 0
        for file_path in dir_path.rglob('*.cache'):
            try:
                if file_path.stat().st_mtime < cutoff_time:
                    file_path.unlink()
                    removed_count += 1
            except (OSError, PermissionError):
                continue

        if removed_count > 0:
            self.logger.info(f"Cleaned up {removed_count} old cache files")

        return removed_count

    def get_disk_usage(self, directory: Union[str, Path]) -> Dict[str, float]:
        """
        Get disk usage statistics for directory

        Args:
            directory: Directory to analyze

        Returns:
            Dictionary with usage statistics (MB)
        """
        dir_path = Path(directory)

        if not dir_path.exists():
            return {'total': 0, 'plt_files': 0, 'cache_files': 0, 'other': 0}

        total_size = 0
        plt_size = 0
        cache_size = 0

        for file_path in dir_path.rglob('*'):
            if file_path.is_file():
                try:
                    size = file_path.stat().st_size
                    total_size += size

                    if 'plt' in file_path.name:
                        plt_size += size
                    elif file_path.suffix in ['.cache', '.pkl']:
                        cache_size += size
                except (OSError, PermissionError):
                    continue

        # Convert to MB
        return {
            'total': total_size / (1024 * 1024),
            'plt_files': plt_size / (1024 * 1024),
            'cache_files': cache_size / (1024 * 1024),
            'other': (total_size - plt_size - cache_size) / (1024 * 1024)
        }

    def _is_plt_directory(self, name: str) -> bool:
        """Check if directory name matches plt pattern"""
        return bool(re.match(r'plt\d+', name))

    def _estimate_time_range(self, sorted_files: List[Path]) -> Tuple[Optional[float], Optional[float]]:
        """Estimate time range from file names (placeholder)"""
        if not sorted_files:
            return None, None

        # This is a placeholder - you'd need to extract actual times
        # from the files or use a naming convention
        return None, None

    def _create_default_logger(self):
        """Create default logger"""
        import logging
        logger = logging.getLogger('FileManager')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger