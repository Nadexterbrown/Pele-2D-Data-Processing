"""
Professional PELE Analysis Package

A comprehensive toolkit for processing and analyzing PELE computational fluid dynamics data.
"""

__version__ = "1.0.0"
__author__ = "Nolan Dexter-Brown"
__email__ = "nadexterbrown@gmail.com"

# Core imports for public API
from .core.data_structures import ProcessedData, WaveData, FlameAnalysis
from .core.io.pele_loader import PeleDataLoader
from .physics.wave_tracking import WaveTracker
from .physics.flame_analysis import FlameAnalyzer
from .workflows.batch_processor import BatchProcessor
from .config.processing_config import ProcessingConfiguration
from .config.thermodynamics import ThermodynamicConditions

# Convenience functions
from .workflows.pipeline_manager import create_processing_pipeline
from .core.io.cache_manager import setup_cache

# Infrastructure
from .infrastructure.logging.mpi_logger import get_logger

__all__ = [
    # Core classes
    'ProcessedData', 'WaveData', 'FlameAnalysis',
    'PeleDataLoader', 'WaveTracker', 'FlameAnalyzer',
    'BatchProcessor', 'ProcessingConfiguration', 'ThermodynamicConditions',

    # Convenience functions
    'create_processing_pipeline', 'setup_cache', 'get_logger',

    # Version info
    '__version__'
]