"""
Core Data Structures for PELE Analysis

Professional data containers that provide type safety, validation,
and clean interfaces for PELE computational fluid dynamics data.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
from pathlib import Path
from enum import Enum
import json

from .exceptions import ValidationError, DataConsistencyError

class ExtractionMethod(Enum):
    """Methods for data extraction from PELE files"""
    AUTO = "auto"
    ORTHO_RAY = "ortho_ray"
    COVERING_GRID = "covering_grid"

class WaveType(Enum):
    """Types of waves that can be tracked"""
    FLAME = "flame"
    SHOCK = "shock"
    DETONATION = "detonation"
    DEFLAGRATION = "deflagration"

@dataclass
class ExtractionParameters:
    """
    Parameters for extracting data from PELE simulations

    Immutable configuration object that defines how data should be
    extracted from PELE plt files.
    """
    location: float                                    # Extraction location (meters)
    direction: str = 'x'                              # Extraction direction ('x' or 'y')

    # Field selection
    fields_requested: Optional[List[str]] = None      # Specific fields (None = all)

    # Method selection
    extraction_method: ExtractionMethod = ExtractionMethod.AUTO

    # Cantera integration
    compute_missing_fields: bool = True               # Use Cantera for missing fields
    mechanism_file: Optional[str] = None              # Cantera mechanism path

    # Performance options
    use_cache: bool = True                           # Enable caching
    parallel_safe: bool = True                       # Safe for MPI parallel use

    # Validation options
    validate_extraction: bool = True                 # Validate extracted data
    tolerance: float = 1e-12                        # Numerical tolerance

    def __post_init__(self):
        """Validate parameters after initialization"""
        self.validate()

    def validate(self) -> bool:
        """
        Validate extraction parameters

        Returns:
            True if valid

        Raises:
            ValidationError: If parameters are invalid
        """
        if self.location < 0:
            raise ValidationError(f"Location must be non-negative, got {self.location}")

        if self.direction not in ['x', 'y']:
            raise ValidationError(f"Direction must be 'x' or 'y', got '{self.direction}'")

        if not isinstance(self.extraction_method, ExtractionMethod):
            try:
                self.extraction_method = ExtractionMethod(self.extraction_method)
            except ValueError:
                raise ValidationError(f"Invalid extraction method: {self.extraction_method}")

        if self.tolerance <= 0:
            raise ValidationError(f"Tolerance must be positive, got {self.tolerance}")

        if self.compute_missing_fields and self.mechanism_file:
            mechanism_path = Path(self.mechanism_file)
            if not mechanism_path.exists():
                raise ValidationError(f"Mechanism file not found: {self.mechanism_file}")

        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'location': self.location,
            'direction': self.direction,
            'fields_requested': self.fields_requested,
            'extraction_method': self.extraction_method.value,
            'compute_missing_fields': self.compute_missing_fields,
            'mechanism_file': self.mechanism_file,
            'use_cache': self.use_cache,
            'parallel_safe': self.parallel_safe,
            'validate_extraction': self.validate_extraction,
            'tolerance': self.tolerance
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExtractionParameters':
        """Create from dictionary"""
        data_copy = data.copy()
        if 'extraction_method' in data_copy:
            data_copy['extraction_method'] = ExtractionMethod(data_copy['extraction_method'])
        return cls(**data_copy)

@dataclass
class ProcessedData:
    """
    Container for processed PELE data with validation and metadata

    Professional data container that ensures consistency and provides
    clean access to extracted simulation data.
    """
    # Core data
    coordinates: np.ndarray                           # Spatial coordinates (SI units)
    fields: Dict[str, np.ndarray]                    # Field data (SI units)

    # Metadata
    source_file: Optional[str] = None                # Source plt file path
    extraction_params: Optional[ExtractionParameters] = None
    timestamp: float = 0.0                           # Simulation time (seconds)

    # Processing info
    processing_metadata: Dict[str, Any] = field(default_factory=dict)
    extraction_time: datetime = field(default_factory=datetime.now)

    # Quality metrics
    data_quality: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        """Validate and process data after initialization"""
        self._validate_data_consistency()
        self._compute_quality_metrics()

    def _validate_data_consistency(self):
        """Validate data consistency"""
        if len(self.coordinates) == 0:
            raise DataConsistencyError("Coordinates array cannot be empty")

        if not self.fields:
            raise DataConsistencyError("Fields dictionary cannot be empty")

        coord_length = len(self.coordinates)
        for field_name, field_data in self.fields.items():
            if not isinstance(field_data, np.ndarray):
                raise DataConsistencyError(f"Field '{field_name}' must be numpy array")

            if len(field_data) != coord_length:
                raise DataConsistencyError(
                    f"Field '{field_name}' length ({len(field_data)}) "
                    f"doesn't match coordinates length ({coord_length})"
                )

            # Check for invalid values
            if np.any(np.isinf(field_data)):
                self.data_quality[f"{field_name}_has_inf"] = True

            if np.any(np.isnan(field_data)):
                nan_count = np.sum(np.isnan(field_data))
                self.data_quality[f"{field_name}_nan_fraction"] = nan_count / len(field_data)

    def _compute_quality_metrics(self):
        """Compute data quality metrics"""
        # Coordinate spacing analysis
        if len(self.coordinates) > 1:
            coord_diff = np.diff(self.coordinates)
            self.data_quality.update({
                'coordinate_spacing_mean': float(np.mean(coord_diff)),
                'coordinate_spacing_std': float(np.std(coord_diff)),
                'coordinate_spacing_uniform': float(np.std(coord_diff) / np.mean(coord_diff))
            })

        # Field statistics
        for field_name, field_data in self.fields.items():
            if np.issubdtype(field_data.dtype, np.number):
                finite_data = field_data[np.isfinite(field_data)]
                if len(finite_data) > 0:
                    self.data_quality[f"{field_name}_range"] = float(np.ptp(finite_data))
                    self.data_quality[f"{field_name}_mean"] = float(np.mean(finite_data))
                    self.data_quality[f"{field_name}_std"] = float(np.std(finite_data))

    def get_field(self, field_name: str, default: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """
        Safely get field data

        Args:
            field_name: Name of field to retrieve
            default: Default value if field not found

        Returns:
            Field data array or default
        """
        return self.fields.get(field_name, default)

    def has_field(self, field_name: str) -> bool:
        """Check if field exists"""
        return field_name in self.fields

    def get_field_names(self) -> List[str]:
        """Get list of available field names"""
        return list(self.fields.keys())

    def add_computed_field(self, name: str, data: np.ndarray,
                          metadata: Optional[Dict] = None) -> None:
        """
        Add computed field with validation

        Args:
            name: Field name
            data: Field data
            metadata: Optional computation metadata

        Raises:
            DataConsistencyError: If data is inconsistent
        """
        if not isinstance(data, np.ndarray):
            data = np.array(data)

        if len(data) != len(self.coordinates):
            raise DataConsistencyError(
                f"New field data length ({len(data)}) must match "
                f"coordinates length ({len(self.coordinates)})"
            )

        self.fields[name] = data

        if metadata:
            self.processing_metadata[f"{name}_computation"] = metadata

        # Update quality metrics for new field
        self._compute_quality_metrics()

    def remove_field(self, field_name: str) -> bool:
        """
        Remove field if it exists

        Args:
            field_name: Name of field to remove

        Returns:
            True if field was removed, False if it didn't exist
        """
        if field_name in self.fields:
            del self.fields[field_name]
            return True
        return False

    def get_coordinate_range(self) -> Tuple[float, float]:
        """Get coordinate range (min, max)"""
        return float(np.min(self.coordinates)), float(np.max(self.coordinates))

    def get_field_range(self, field_name: str) -> Optional[Tuple[float, float]]:
        """
        Get field value range

        Args:
            field_name: Field to analyze

        Returns:
            (min, max) tuple or None if field doesn't exist
        """
        if not self.has_field(field_name):
            return None

        field_data = self.fields[field_name]
        finite_data = field_data[np.isfinite(field_data)]

        if len(finite_data) == 0:
            return None

        return float(np.min(finite_data)), float(np.max(finite_data))

    def interpolate_at_location(self, location: float, field_name: str,
                               method: str = 'linear') -> Optional[float]:
        """
        Interpolate field value at specific location

        Args:
            location: Location for interpolation
            field_name: Field to interpolate
            method: Interpolation method

        Returns:
            Interpolated value or None if field doesn't exist
        """
        if not self.has_field(field_name):
            return None

        from scipy.interpolate import interp1d

        try:
            field_data = self.fields[field_name]
            # Remove NaN/inf values for interpolation
            valid_mask = np.isfinite(field_data)

            if np.sum(valid_mask) < 2:
                return None  # Need at least 2 points for interpolation

            valid_coords = self.coordinates[valid_mask]
            valid_data = field_data[valid_mask]

            interpolator = interp1d(valid_coords, valid_data,
                                  kind=method, bounds_error=False,
                                  fill_value=np.nan)

            result = interpolator(location)
            return float(result) if np.isfinite(result) else None

        except Exception:
            return None

    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive data summary"""
        coord_min, coord_max = self.get_coordinate_range()

        field_summaries = {}
        for field_name in self.fields:
            field_range = self.get_field_range(field_name)
            field_summaries[field_name] = {
                'range': field_range,
                'has_nans': f"{field_name}_nan_fraction" in self.data_quality,
                'has_infs': f"{field_name}_has_inf" in self.data_quality
            }

        return {
            'source_file': Path(self.source_file).name if self.source_file else None,
            'num_points': len(self.coordinates),
            'coordinate_range': {'min': coord_min, 'max': coord_max},
            'fields': field_summaries,
            'simulation_time': self.timestamp,
            'extraction_time': self.extraction_time.isoformat(),
            'data_quality': self.data_quality,
            'processing_info': self.processing_metadata
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'coordinates': self.coordinates.tolist(),
            'fields': {name: data.tolist() for name, data in self.fields.items()},
            'source_file': self.source_file,
            'extraction_params': self.extraction_params.to_dict() if self.extraction_params else None,
            'timestamp': self.timestamp,
            'processing_metadata': self.processing_metadata,
            'extraction_time': self.extraction_time.isoformat(),
            'data_quality': self.data_quality
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProcessedData':
        """Create from dictionary"""
        coords = np.array(data['coordinates'])
        fields = {name: np.array(values) for name, values in data['fields'].items()}

        extraction_params = None
        if data.get('extraction_params'):
            extraction_params = ExtractionParameters.from_dict(data['extraction_params'])

        extraction_time = datetime.now()
        if data.get('extraction_time'):
            extraction_time = datetime.fromisoformat(data['extraction_time'])

        return cls(
            coordinates=coords,
            fields=fields,
            source_file=data.get('source_file'),
            extraction_params=extraction_params,
            timestamp=data.get('timestamp', 0.0),
            processing_metadata=data.get('processing_metadata', {}),
            extraction_time=extraction_time,
            data_quality=data.get('data_quality', {})
        )

@dataclass
class WaveData:
    """
    Container for wave tracking results

    Represents detected waves (flames, shocks, etc.) with position,
    velocity, and associated properties.
    """
    wave_type: WaveType                              # Type of wave
    index: int                                       # Array index of wave position
    position: float                                  # Physical position (meters)

    # Optional properties
    velocity: Optional[float] = None                 # Wave velocity (m/s)
    acceleration: Optional[float] = None             # Wave acceleration (m/s²)

    # Wave properties
    properties: Dict[str, float] = field(default_factory=dict)

    # Quality metrics
    confidence: float = 1.0                         # Detection confidence (0-1)
    detection_method: str = "threshold"             # Detection method used

    # Metadata
    detection_time: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Validate wave data after initialization"""
        if not isinstance(self.wave_type, WaveType):
            try:
                self.wave_type = WaveType(self.wave_type)
            except ValueError:
                raise ValidationError(f"Invalid wave type: {self.wave_type}")

        if not 0 <= self.confidence <= 1:
            raise ValidationError(f"Confidence must be between 0 and 1, got {self.confidence}")

    def is_valid(self) -> bool:
        """Check if wave detection is valid"""
        return (
            self.index >= 0 and
            not np.isnan(self.position) and
            self.confidence > 0
        )

    def add_property(self, name: str, value: float, description: str = ""):
        """Add wave property with optional description"""
        self.properties[name] = value
        if description:
            self.properties[f"{name}_description"] = description

    def get_property(self, name: str, default: float = np.nan) -> float:
        """Get wave property with default"""
        return self.properties.get(name, default)

    @classmethod
    def invalid(cls, wave_type: Union[WaveType, str], reason: str = "Detection failed") -> 'WaveData':
        """Create invalid wave data for error cases"""
        if isinstance(wave_type, str):
            wave_type = WaveType(wave_type)

        return cls(
            wave_type=wave_type,
            index=-1,
            position=np.nan,
            confidence=0.0,
            detection_method=reason
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'wave_type': self.wave_type.value,
            'index': self.index,
            'position': self.position,
            'velocity': self.velocity,
            'acceleration': self.acceleration,
            'properties': self.properties,
            'confidence': self.confidence,
            'detection_method': self.detection_method,
            'detection_time': self.detection_time.isoformat(),
            'valid': self.is_valid()
        }

@dataclass
class FlameAnalysis:
    """
    Comprehensive flame analysis results

    Contains all flame-related measurements and calculations including
    position, geometry, thermodynamics, and combustion properties.
    """
    # Core flame data
    position_data: WaveData                         # Flame position information

    # Geometric properties
    thickness: Optional[float] = None               # Flame thickness (meters)
    surface_length: Optional[float] = None          # Flame surface length (meters)
    curvature: Optional[float] = None              # Mean curvature (1/meters)

    # Combustion properties
    consumption_rate: Optional[float] = None        # Fuel consumption rate (kg/s)
    burning_velocity: Optional[float] = None        # Laminar burning velocity (m/s)
    heat_release_rate: Optional[float] = None       # Heat release rate (W/m³)

    # Thermodynamic state
    thermodynamic_state: Optional[Dict[str, float]] = None

    # Flow properties
    reynolds_number: Optional[float] = None         # Local Reynolds number
    damkohler_number: Optional[float] = None       # Damköhler number

    # Analysis metadata
    geometry_metadata: Dict[str, Any] = field(default_factory=dict)
    analysis_time: datetime = field(default_factory=datetime.now)

    def is_valid(self) -> bool:
        """Check if flame analysis is valid"""
        return self.position_data.is_valid()

    def get_flame_position(self) -> float:
        """Get flame position"""
        return self.position_data.position

    def get_flame_velocity(self) -> Optional[float]:
        """Get flame velocity"""
        return self.position_data.velocity

    def add_thermodynamic_property(self, name: str, value: float):
        """Add thermodynamic property"""
        if self.thermodynamic_state is None:
            self.thermodynamic_state = {}
        self.thermodynamic_state[name] = value

    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive flame analysis summary"""
        return {
            'position': self.position_data.position if self.is_valid() else None,
            'velocity': self.position_data.velocity,
            'thickness': self.thickness,
            'surface_length': self.surface_length,
            'consumption_rate': self.consumption_rate,
            'burning_velocity': self.burning_velocity,
            'heat_release_rate': self.heat_release_rate,
            'thermodynamic_state': self.thermodynamic_state,
            'reynolds_number': self.reynolds_number,
            'valid': self.is_valid(),
            'analysis_time': self.analysis_time.isoformat()
        }

@dataclass
class ProcessingResults:
    """
    Complete processing results for a single simulation file

    Aggregates all analysis results for a plt file including timing,
    flame data, shock data, and processing statistics.
    """
    # File information
    file_path: str                                  # Source file path
    processing_time: float                          # Processing duration (seconds)
    simulation_time: float                          # Simulation timestamp

    # Analysis results
    flame_analysis: Optional[FlameAnalysis] = None
    shock_analysis: Optional[WaveData] = None
    extracted_data: Optional[ProcessedData] = None

    # Processing statistics
    processing_stats: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)

    def add_error(self, error: str):
        """Add processing error"""
        self.errors.append(error)

    def add_warning(self, warning: str):
        """Add processing warning"""
        self.warnings.append(warning)

    def has_errors(self) -> bool:
        """Check if processing had errors"""
        return len(self.errors) > 0

    def has_warnings(self) -> bool:
        """Check if processing had warnings"""
        return len(self.warnings) > 0

    def is_successful(self) -> bool:
        """Check if processing was successful"""
        return not self.has_errors() and (
            self.flame_analysis is not None or
            self.shock_analysis is not None or
            self.extracted_data is not None
        )

    def get_summary(self) -> Dict[str, Any]:
        """Get processing summary"""
        return {
            'file': Path(self.file_path).name,
            'simulation_time': self.simulation_time,
            'processing_time': self.processing_time,
            'successful': self.is_successful(),
            'has_flame_data': self.flame_analysis is not None,
            'has_shock_data': self.shock_analysis is not None,
            'has_extracted_data': self.extracted_data is not None,
            'errors': len(self.errors),
            'warnings': len(self.warnings),
            'created_at': self.created_at.isoformat()
        }