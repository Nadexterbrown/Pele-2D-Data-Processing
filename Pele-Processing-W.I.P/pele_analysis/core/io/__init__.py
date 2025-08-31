"""
Core I/O Module for PELE Analysis

This module provides all input/output functionality for PELE data processing,
including data loading, file management, caching, and output writing.
"""

from .pele_loader import PeleDataLoader, ExtractedData, ExtractionParameters
from .file_manager import FileManager, DirectoryInfo
from .output_writer import OutputWriter, OutputFormat
from .cache_manager import create_cache_manager, CacheConfig, ParallelStrategy

__all__ = [
    'PeleDataLoader',
    'ExtractedData',
    'ExtractionParameters',
    'FileManager',
    'DirectoryInfo',
    'OutputWriter',
    'OutputFormat',
    'create_cache_manager',
    'CacheConfig',
    'ParallelStrategy'
]