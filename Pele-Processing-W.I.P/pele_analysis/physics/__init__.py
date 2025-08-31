"""
Physics Module for PELE Analysis

Physics calculations for combustion and fluid dynamics,
including wave tracking, flame analysis, and thermodynamic calculations.
"""

from .wave_tracking import WaveTracker, WaveDetectionMethod
from .flame_analysis import FlameAnalyzer, FlameGeometryAnalyzer
from .thermodynamics import ThermodynamicCalculator, CanteraBridge
from .combustion import CombustionAnalyzer, BurningVelocityCalculator

__all__ = [
    'WaveTracker',
    'WaveDetectionMethod',
    'FlameAnalyzer',
    'FlameGeometryAnalyzer',
    'ThermodynamicCalculator',
    'CanteraBridge',
    'CombustionAnalyzer',
    'BurningVelocityCalculator'
]