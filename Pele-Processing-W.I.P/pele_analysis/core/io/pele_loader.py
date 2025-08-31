"""
Professional PELE Data Extraction Module

This module provides clean, efficient data extraction from PELE simulation results.
Replaces the tangled data_ray_extraction function with proper separation of concerns.

Author: Your Name
Created: 2024
"""

import os
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import yt
import cantera as ct

# Suppress yt warnings for cleaner output
yt.set_log_level(50)


# ===================================================================
# DATA STRUCTURES
# ===================================================================

@dataclass
class ExtractionParameters:
    """Parameters for data extraction from PELE simulations"""

    # Core extraction parameters
    location: float  # Extraction location (meters)
    direction: str = 'x'  # Extraction direction ('x' or 'y')

    # Field selection
    fields_requested: Optional[List[str]] = None  # Specific fields to extract (None = all available)

    # Performance options
    use_cache: bool = True  # Enable result caching
    extraction_method: str = 'auto'  # 'auto', 'ortho_ray', 'covering_grid'

    # Cantera options for missing field computation
    compute_missing_fields: bool = True  # Compute missing fields with Cantera
    mechanism_file: Optional[str] = None  # Cantera mechanism file path

    def validate(self) -> bool:
        """Validate extraction parameters"""
        if self.location < 0:
            raise ValueError(f"Location must be non-negative, got {self.location}")

        if self.direction not in ['x', 'y']:
            raise ValueError(f"Direction must be 'x' or 'y', got '{self.direction}'")

        if self.extraction_method not in ['auto', 'ortho_ray', 'covering_grid']:
            raise ValueError(f"Invalid extraction method: {self.extraction_method}")

        if self.compute_missing_fields and not self.mechanism_file:
            warnings.warn("Missing fields computation requested but no mechanism file provided")

        return True


@dataclass
class ExtractedData:
    """Container for extracted PELE data with metadata"""

    # Core data
    coordinates: np.ndarray  # Spatial coordinates (meters)
    fields: Dict[str, np.ndarray]  # Field data

    # Metadata
    source_file: Optional[str] = None  # Source file path
    extraction_params: Optional[ExtractionParameters] = None
    timestamp: float = 0.0  # Simulation time
    extraction_time: datetime = field(default_factory=datetime.now)

    # Processing metadata
    processing_info: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate data consistency after initialization"""
        if len(self.coordinates) == 0:
            raise ValueError("Coordinates array cannot be empty")

        # Validate all fields have same length as coordinates
        coord_length = len(self.coordinates)
        for field_name, field_data in self.fields.items():
            if len(field_data) != coord_length:
                raise ValueError(
                    f"Field '{field_name}' length ({len(field_data)}) "
                    f"doesn't match coordinates length ({coord_length})"
                )

    def get_field(self, field_name: str, default: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """Safely get field data"""
        return self.fields.get(field_name, default)

    def has_field(self, field_name: str) -> bool:
        """Check if field exists"""
        return field_name in self.fields

    def add_computed_field(self, name: str, data: np.ndarray, metadata: Optional[Dict] = None):
        """Add a computed field with validation"""
        if len(data) != len(self.coordinates):
            raise ValueError(f"Field data length must match coordinates ({len(self.coordinates)})")

        self.fields[name] = data
        if metadata:
            self.processing_info[f"{name}_computation"] = metadata

    def get_summary(self) -> Dict[str, Any]:
        """Get extraction summary for logging/debugging"""
        return {
            'source_file': Path(self.source_file).name if self.source_file else None,
            'num_points': len(self.coordinates),
            'coordinate_range': {
                'min': float(np.min(self.coordinates)),
                'max': float(np.max(self.coordinates)),
                'span': float(np.max(self.coordinates) - np.min(self.coordinates))
            },
            'available_fields': list(self.fields.keys()),
            'timestamp': self.timestamp,
            'extraction_location': self.extraction_params.location if self.extraction_params else None,
            'extraction_time': self.extraction_time.isoformat()
        }


# ===================================================================
# FIELD DEFINITIONS
# ===================================================================

@dataclass(frozen=True)
class FieldDefinition:
    """Definition of a PELE field"""
    pele_name: str  # Name in PELE output
    units: str  # Physical units
    description: str  # Human-readable description
    cantera_computable: bool = False  # Can be computed with Cantera
    required: bool = False  # Required for basic analysis


# Clean field definitions with proper organization
PELE_FIELDS = {
    # Coordinates
    'X': FieldDefinition('x', 'cm', 'X coordinate'),
    'Y': FieldDefinition('y', 'cm', 'Y coordinate'),

    # Basic thermodynamic properties
    'Temperature': FieldDefinition('Temp', 'K', 'Temperature'),
    'Pressure': FieldDefinition('pressure', 'g / cm / s^2', 'Pressure'),
    'Density': FieldDefinition('density', 'g / cm^3', 'Density', cantera_computable=True),

    # Transport properties
    'Viscosity': FieldDefinition('viscosity', 'g / cm / s', 'Dynamic viscosity', cantera_computable=True),
    'Conductivity': FieldDefinition('conductivity', 'g cm^2 / s^3 / cm / K', 'Thermal conductivity',
                                    cantera_computable=True),
    'Sound_Speed': FieldDefinition('soundspeed', 'cm / s', 'Speed of sound', cantera_computable=True),

    # Flow properties
    'X_Velocity': FieldDefinition('x_velocity', 'cm / s', 'X-direction velocity'),
    'Y_Velocity': FieldDefinition('y_velocity', 'cm / s', 'Y-direction velocity'),
    'Mach_Number': FieldDefinition('MachNumber', '', 'Mach number', cantera_computable=True),

    # Combustion properties
    'Heat_Release_Rate': FieldDefinition('heatRelease', 'g cm^2 / s^3 / cm^3', 'Heat release rate'),
    'Cp': FieldDefinition('cp', 'g cm^2 / s^2 / g / K', 'Specific heat (constant pressure)', cantera_computable=True),
    'Cv': FieldDefinition('cv', 'g cm^2 / s^2 / g / K', 'Specific heat (constant volume)', cantera_computable=True),
}


def add_species_fields(species_list: List[str]) -> None:
    """Add species-specific fields to global registry"""
    for species in species_list:
        # Mass fraction
        PELE_FIELDS[f'Y_{species}'] = FieldDefinition(
            f'Y({species})', '', f'{species} mass fraction'
        )
        # Diffusion coefficient
        PELE_FIELDS[f'D_{species}'] = FieldDefinition(
            f'D({species})', 'cm^2 / s', f'{species} diffusion coefficient', cantera_computable=True
        )
        # Production rate
        PELE_FIELDS[f'W_{species}'] = FieldDefinition(
            f'rho_omega_{species}', 'g / cm^3 / s', f'{species} production rate'
        )


# ===================================================================
# UNIT CONVERSION
# ===================================================================

class UnitConverter:
    """Clean unit conversion with caching"""

    BASE_CONVERSIONS = {
        's': 1, 'g': 1e-3, 'mol': 1e-3, 'cm': 1e-2,
        'K': 1, 'kg': 1, 'm': 1, 'Pa': 1
    }

    def __init__(self):
        self._conversion_cache: Dict[str, float] = {}

    def convert_to_si(self, value: Union[float, np.ndarray], field_name: str) -> Union[float, np.ndarray]:
        """Convert field to SI units"""

        if field_name not in PELE_FIELDS:
            return value  # Unknown field, return as-is

        units = PELE_FIELDS[field_name].units
        if not units.strip():
            return value  # Dimensionless

        # Get conversion factor (cached)
        if field_name not in self._conversion_cache:
            self._conversion_cache[field_name] = self._compute_conversion_factor(units)

        conversion_factor = self._conversion_cache[field_name]
        return value * conversion_factor

    def _compute_conversion_factor(self, unit_string: str) -> float:
        """Compute conversion factor from PELE units to SI"""

        if not unit_string.strip():
            return 1.0

        try:
            # Handle compound units like "g / cm / s^2"
            if '/' in unit_string:
                parts = unit_string.split('/')
                numerator = parts[0].strip()
                denominator_parts = [p.strip() for p in parts[1:]]

                num_factor = self._parse_unit_part(numerator)
                den_factor = 1.0
                for part in denominator_parts:
                    den_factor *= self._parse_unit_part(part)

                return num_factor / den_factor
            else:
                return self._parse_unit_part(unit_string.strip())

        except Exception as e:
            warnings.warn(f"Could not parse units '{unit_string}': {e}")
            return 1.0

    def _parse_unit_part(self, unit_part: str) -> float:
        """Parse individual unit part with optional exponent"""

        import re

        if not unit_part:
            return 1.0

        # Handle exponents like "cm^2" or "s^-1"
        match = re.match(r'([a-zA-Z]+)(?:\^(-?\d+))?', unit_part)
        if not match:
            return 1.0

        base_unit, exponent = match.groups()
        exp_value = int(exponent) if exponent else 1

        if base_unit in self.BASE_CONVERSIONS:
            return self.BASE_CONVERSIONS[base_unit] ** exp_value

        return 1.0


# ===================================================================
# MAIN DATA LOADER CLASS
# ===================================================================

class PeleDataLoader:
    """
    PELE data loader with caching and error handling.
    """

    def __init__(self, cache_manager=None, logger=None):
        """
        Initialize PELE data loader

        Args:
            cache_manager: Optional cache manager for results
            logger: Optional logger instance
        """
        self.cache_manager = cache_manager
        self.logger = logger or self._create_default_logger()
        self.unit_converter = UnitConverter()

        # Track currently loaded dataset for proper cleanup
        self._current_dataset = None
        self._cantera_gas = None

    @contextmanager
    def managed_dataset(self, file_path: Union[str, Path]):
        """
        Context manager for safe yt dataset loading and cleanup

        This ensures proper memory management and prevents dataset leaks
        """
        file_path = Path(file_path)
        dataset = None
        load_start = time.time()

        try:
            self.logger.info(f"Loading dataset: {file_path.name}")

            # Load dataset
            dataset = yt.load(str(file_path))
            self._current_dataset = dataset

            load_time = time.time() - load_start
            if load_time > 2.0:
                self.logger.warning(f"Slow dataset load: {load_time:.2f}s for {file_path.name}")

            yield dataset

        except Exception as e:
            self.logger.error(f"Failed to load dataset {file_path.name}: {e}")
            raise RuntimeError(f"Dataset loading failed: {e}") from e

        finally:
            # Aggressive cleanup to prevent memory leaks
            if dataset:
                self._cleanup_dataset(dataset)
            self._current_dataset = None

    def extract_ray_data(self, file_path: Union[str, Path],
                         params: ExtractionParameters) -> ExtractedData:
        """
        Extract 1D ray data from PELE simulation results

        Args:
            file_path: Path to PELE plt file
            params: Extraction parameters

        Returns:
            ExtractedData: Processed data with metadata

        Raises:
            ValueError: Invalid parameters
            RuntimeError: Extraction failed
        """
        file_path = Path(file_path)
        params.validate()

        # Check cache first
        if self.cache_manager and params.use_cache:
            cached = self.cache_manager.get_cached_data(file_path, params)
            if cached is not None:
                self.logger.info(f"Cache hit: {file_path.name}")
                return cached

        # Extract data from file
        extraction_start = time.time()

        try:
            with self.managed_dataset(file_path) as dataset:
                extracted_data = self._perform_extraction(dataset, params)
                extracted_data.source_file = str(file_path)
                extracted_data.extraction_params = params

                # Add processing metadata
                extraction_time = time.time() - extraction_start
                extracted_data.processing_info.update({
                    'extraction_time_seconds': extraction_time,
                    'dataset_domain_size': tuple(dataset.domain_dimensions),
                    'max_refinement_level': dataset.index.max_level
                })

                # Cache result if enabled
                if self.cache_manager and params.use_cache:
                    self.cache_manager.cache_data(file_path, params, extracted_data)

                self.logger.info(
                    f"Extracted {len(extracted_data.coordinates)} points from {file_path.name} in {extraction_time:.2f}s")
                return extracted_data

        except Exception as e:
            self.logger.error(f"Extraction failed for {file_path.name}: {e}")
            raise RuntimeError(f"Data extraction failed: {e}") from e

    def _perform_extraction(self, dataset, params: ExtractionParameters) -> ExtractedData:
        """Perform the actual data extraction from yt dataset"""

        # Determine extraction method
        if params.extraction_method == 'auto':
            method = 'ortho_ray' if hasattr(dataset, 'ortho_ray') else 'covering_grid'
        else:
            method = params.extraction_method

        self.logger.debug(f"Using extraction method: {method}")

        # Extract data
        if method == 'ortho_ray':
            raw_data = self._extract_via_ortho_ray(dataset, params)
        else:
            raw_data = self._extract_via_covering_grid(dataset, params)

        # Convert units to SI
        converted_data = self._convert_units(raw_data)

        # Compute missing fields if requested
        if params.compute_missing_fields and params.mechanism_file:
            self._compute_missing_fields(converted_data, params.mechanism_file)

        return converted_data

    def _extract_via_ortho_ray(self, dataset, params: ExtractionParameters) -> ExtractedData:
        """Extract data using yt ortho_ray method (preferred for 1D extractions)"""

        # Create ray
        if params.direction == 'x':
            ray = dataset.ortho_ray(0, (params.location * 100, 0))  # Convert location to cm
        else:
            ray = dataset.ortho_ray(1, (0, params.location * 100))

        # Sort by coordinate
        sort_indices = np.argsort(ray['boxlib','x'])
        coordinates = ray['boxlib','x'][sort_indices].to_value() / 100  # Convert to meters

        # Extract available fields
        fields = {}
        available_dataset_fields = {field[1] for field in dataset.field_list if field[0] == 'boxlib'}
        available_dataset_fields.add('Temp')  # yt special case

        for field_key, field_def in PELE_FIELDS.items():
            # Skip if specific fields requested and this isn't one
            if params.fields_requested and field_key not in params.fields_requested:
                continue

            pele_name = field_def.pele_name

            try:
                # Handle special yt fields
                if pele_name == 'Temp':
                    raw_data = ray['boxlib', 'Temp'][sort_indices].to_value()
                elif pele_name in available_dataset_fields:
                    raw_data = ray['boxlib', pele_name][sort_indices].to_value()
                else:
                    continue  # Field not available

                fields[field_key] = raw_data
                self.logger.debug(f"Extracted field: {field_key}")

            except Exception as e:
                self.logger.warning(f"Could not extract field {field_key}: {e}")
                continue

        return ExtractedData(
            coordinates=coordinates,
            fields=fields,
            timestamp=dataset.current_time.to_value(),
            processing_info={'extraction_method': 'ortho_ray'}
        )

    def _extract_via_covering_grid(self, dataset, params: ExtractionParameters) -> ExtractedData:
        """Extract data using yt covering_grid method (fallback method)"""

        max_level = dataset.index.max_level

        # Calculate extraction coordinates in dataset units (cm)
        location_cm = params.location * 100
        buffer_cm = dataset.index.get_smallest_dx().to_value() / 2

        if params.direction == 'x':
            left_edge = [dataset.domain_left_edge[0].to_value(), location_cm - buffer_cm, 0.0]
            right_edge = [dataset.domain_right_edge[0].to_value(), location_cm + buffer_cm,
                          dataset.domain_right_edge[2].to_value()]
            dims = [dataset.domain_dimensions[0] * 2 ** max_level, 1, 1]
        else:
            left_edge = [location_cm - buffer_cm, dataset.domain_left_edge[1].to_value(), 0.0]
            right_edge = [location_cm + buffer_cm, dataset.domain_right_edge[1].to_value(),
                          dataset.domain_right_edge[2].to_value()]
            dims = [1, dataset.domain_dimensions[1] * 2 ** max_level, 1]

        # Create covering grid
        covering_grid = dataset.covering_grid(
            level=max_level,
            left_edge=left_edge,
            dims=dims
        )

        # Extract coordinates
        if params.direction == 'x':
            coordinates = covering_grid['boxlib', 'x'][:, 0, 0].to_value() / 100  # Convert to meters
        else:
            coordinates = covering_grid['boxlib', 'y'][0, :, 0].to_value() / 100

        # Extract fields
        fields = {}
        available_fields = {field[1] for field in dataset.field_list if field[0] == 'boxlib'}

        for field_key, field_def in PELE_FIELDS.items():
            if params.fields_requested and field_key not in params.fields_requested:
                continue

            pele_name = field_def.pele_name

            try:
                if pele_name == 'Temp':
                    if params.direction == 'x':
                        raw_data = covering_grid['boxlib', 'Temp'][:, 0, 0].to_value()
                    else:
                        raw_data = covering_grid['boxlib', 'Temp'][0, :, 0].to_value()
                elif pele_name in available_fields:
                    if params.direction == 'x':
                        raw_data = covering_grid['boxlib', pele_name][:, 0, 0].to_value()
                    else:
                        raw_data = covering_grid['boxlib', pele_name][0, :, 0].to_value()
                else:
                    continue

                fields[field_key] = raw_data

            except Exception as e:
                self.logger.warning(f"Could not extract field {field_key}: {e}")
                continue

        return ExtractedData(
            coordinates=coordinates,
            fields=fields,
            timestamp=dataset.current_time.to_value(),
            processing_info={'extraction_method': 'covering_grid'}
        )

    def _convert_units(self, data: ExtractedData) -> ExtractedData:
        """Convert all fields from PELE units to SI units"""

        converted_fields = {}

        for field_name, field_data in data.fields.items():
            try:
                converted_data = self.unit_converter.convert_to_si(field_data, field_name)
                converted_fields[field_name] = converted_data
            except Exception as e:
                self.logger.warning(f"Unit conversion failed for {field_name}: {e}")
                converted_fields[field_name] = field_data  # Use original data

        # Update data with converted fields
        data.fields = converted_fields
        data.processing_info['units_converted_to_si'] = True

        return data

    def _compute_missing_fields(self, data: ExtractedData, mechanism_file: str):
        """Compute missing fields using Cantera"""

        try:
            # Initialize Cantera gas if needed
            if self._cantera_gas is None:
                self._cantera_gas = ct.Solution(mechanism_file)

            gas = self._cantera_gas

            # Check for required fields
            if not (data.has_field('Temperature') and data.has_field('Pressure')):
                self.logger.warning("Temperature and Pressure required for Cantera calculations")
                return

            temp_data = data.get_field('Temperature')
            pressure_data = data.get_field('Pressure')

            # Get species mass fractions
            species_data = {}
            for field_name in data.fields:
                if field_name.startswith('Y_'):
                    species_name = field_name[2:]  # Remove 'Y_' prefix
                    species_data[species_name] = data.get_field(field_name)

            # Find missing computable fields
            missing_fields = []
            for field_name, field_def in PELE_FIELDS.items():
                if field_def.cantera_computable and not data.has_field(field_name):
                    missing_fields.append(field_name)

            if not missing_fields:
                return

            self.logger.info(f"Computing {len(missing_fields)} missing fields with Cantera")

            # Compute missing fields point by point
            computed_data = {field: [] for field in missing_fields}

            for i in range(len(temp_data)):
                T = temp_data[i]
                P = pressure_data[i]

                # Set species composition
                if species_data:
                    Y_dict = {species: species_data[species][i] for species in species_data}
                    try:
                        gas.TPY = T, P, Y_dict
                    except Exception as e:
                        # Use default composition if species data is problematic
                        self.logger.warning(f"Cantera composition error at point {i}: {e}")
                        gas.TP = T, P
                else:
                    gas.TP = T, P

                # Compute requested properties
                for field_name in missing_fields:
                    try:
                        if field_name == 'Density':
                            computed_data[field_name].append(gas.density_mass)
                        elif field_name == 'Viscosity':
                            computed_data[field_name].append(gas.viscosity)
                        elif field_name == 'Conductivity':
                            computed_data[field_name].append(gas.thermal_conductivity)
                        elif field_name == 'Sound_Speed':
                            computed_data[field_name].append(gas.sound_speed)
                        elif field_name == 'Cp':
                            computed_data[field_name].append(gas.cp_mass)
                        elif field_name == 'Cv':
                            computed_data[field_name].append(gas.cv_mass)
                        elif field_name == 'Mach_Number':
                            if data.has_field('X_Velocity'):
                                velocity = data.get_field('X_Velocity')[i]
                                computed_data[field_name].append(velocity / gas.sound_speed)
                            else:
                                computed_data[field_name].append(np.nan)
                        else:
                            computed_data[field_name].append(np.nan)
                    except Exception as e:
                        computed_data[field_name].append(np.nan)

            # Add computed fields to data
            for field_name, values in computed_data.items():
                if any(not np.isnan(v) for v in values):  # Only add if some values are valid
                    data.add_computed_field(
                        field_name,
                        np.array(values),
                        {'computation_method': 'cantera', 'mechanism_file': mechanism_file}
                    )

            data.processing_info['cantera_computation_completed'] = True

        except Exception as e:
            self.logger.error(f"Cantera computation failed: {e}")
            data.processing_info['cantera_computation_failed'] = str(e)

    def _cleanup_dataset(self, dataset):
        """Aggressive cleanup of yt dataset to prevent memory leaks"""
        try:
            if hasattr(dataset, 'index'):
                del dataset.index
            if hasattr(dataset, '_hash'):
                del dataset._hash
            if hasattr(dataset, 'field_list'):
                del dataset.field_list
        except:
            pass

        try:
            del dataset
        except:
            pass

        # Force garbage collection
        import gc
        gc.collect()

    def _create_default_logger(self):
        """Create default logger if none provided"""
        import logging

        logger = logging.getLogger('PeleDataLoader')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)

        return logger


# ===================================================================
# CONVENIENCE FUNCTIONS
# ===================================================================

def extract_pele_data(file_path: Union[str, Path], location: float,
                      direction: str = 'x', mechanism_file: Optional[str] = None,
                      fields: Optional[List[str]] = None, species_list: Optional[List[str]] = None) -> ExtractedData:
    """
    Convenience function for quick data extraction

    Args:
        file_path: Path to PELE plt file
        location: Extraction location in meters
        direction: Extraction direction ('x' or 'y')
        mechanism_file: Optional Cantera mechanism file for missing field computation
        fields: Optional list of specific fields to extract
        species_list: Optional list of species for field definitions

    Returns:
        ExtractedData: Extracted and processed data
    """

    # Add species fields if provided
    if species_list:
        add_species_fields(species_list)

    # Create extraction parameters
    params = ExtractionParameters(
        location=location,
        direction=direction,
        fields_requested=fields,
        mechanism_file=mechanism_file,
        compute_missing_fields=mechanism_file is not None
    )

    # Extract data
    loader = PeleDataLoader()
    return loader.extract_ray_data(file_path, params)


# ===================================================================
# USAGE EXAMPLE
# ===================================================================

if __name__ == "__main__":
    """Example usage of the PELE data loader"""

    # Example: Extract data at 5cm with H2 combustion mechanism
    try:
        data = extract_pele_data(
            file_path="../../../../2D-Test-Data/plt73000",
            location=0.0445 / 100,  # 5cm
            direction='x',
            mechanism_file="../../mechanism_files/LiDryer.yaml",
            species_list=['H2', 'O2', 'N2', 'H2O', 'OH', 'HO2']
        )

        print("Extraction successful!")
        print(f"Extracted {len(data.coordinates)} points")
        print(f"Available fields: {list(data.fields.keys())}")

        # Access specific field data
        if data.has_field('Temperature'):
            temp = data.get_field('Temperature')
            print(f"Temperature range: {np.min(temp):.1f} - {np.max(temp):.1f} K")

        # Print summary
        import json

        print(json.dumps(data.get_summary(), indent=2))

    except Exception as e:
        print(f"Extraction failed: {e}")