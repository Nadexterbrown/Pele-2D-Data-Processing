"""
Thermodynamic Calculations for PELE Analysis

Professional thermodynamic property calculations using Cantera integration
for missing field computation and validation.
"""

import numpy as np
import cantera as ct
from typing import Dict, List, Optional, Any, Union
from ..core.data_structures import ProcessedData
from ..core.exceptions import ThermodynamicsError, ValidationError

class CanteraBridge:
    """
    Professional Cantera integration for thermodynamic calculations

    Provides clean interface to Cantera for computing missing thermodynamic
    and transport properties in PELE simulations.
    """

    def __init__(self, mechanism_file: str, logger=None):
        self.mechanism_file = mechanism_file
        self.logger = logger or self._create_default_logger()

        try:
            self.gas = ct.Solution(mechanism_file)
            self.species_names = self.gas.species_names
            self.logger.info(f"Loaded mechanism: {mechanism_file} ({len(self.species_names)} species)")
        except Exception as e:
            raise ThermodynamicsError(f"Failed to load Cantera mechanism: {e}")

    def compute_missing_properties(self, data: ProcessedData,
                                  missing_fields: List[str]) -> Dict[str, np.ndarray]:
        """
        Compute missing thermodynamic properties using Cantera

        Args:
            data: Processed data with temperature, pressure, and species
            missing_fields: List of fields to compute

        Returns:
            Dictionary of computed properties

        Raises:
            ThermodynamicsError: If computation fails
        """
        if not data.has_field('Temperature') or not data.has_field('Pressure'):
            raise ThermodynamicsError("Temperature and Pressure required for Cantera calculations")

        temp_data = data.get_field('Temperature')
        pressure_data = data.get_field('Pressure')

        # Extract species data
        species_data = self._extract_species_data(data)

        # Prepare output arrays
        computed_props = {field: np.full(len(temp_data), np.nan) for field in missing_fields}

        # Process point by point
        successful_computations = 0

        for i in range(len(temp_data)):
            try:
                T = temp_data[i]
                P = pressure_data[i]

                # Validate thermodynamic state
                if not self._validate_state(T, P):
                    continue

                # Set species composition
                Y_dict = {species: species_data[species][i] for species in species_data}

                if Y_dict:
                    self.gas.TPY = T, P, Y_dict
                else:
                    self.gas.TP = T, P

                # Compute requested properties
                for field in missing_fields:
                    computed_props[field][i] = self._compute_property(field)

                successful_computations += 1

            except Exception as e:
                self.logger.debug(f"Cantera computation failed at point {i}: {e}")
                continue

        success_rate = successful_computations / len(temp_data)
        self.logger.info(f"Cantera computation success rate: {success_rate:.1%}")

        if success_rate < 0.5:
            self.logger.warning("Low Cantera computation success rate - check thermodynamic states")

        return computed_props

    def _extract_species_data(self, data: ProcessedData) -> Dict[str, np.ndarray]:
        """Extract species mass fractions from data"""
        species_data = {}

        for field_name in data.get_field_names():
            if field_name.startswith('Y_'):
                species_name = field_name[2:]  # Remove 'Y_' prefix
                if species_name in self.species_names:
                    species_data[species_name] = data.get_field(field_name)

        return species_data

    def _validate_state(self, temperature: float, pressure: float) -> bool:
        """Validate thermodynamic state for Cantera"""
        return (
            np.isfinite(temperature) and np.isfinite(pressure) and
            temperature > 200 and temperature < 6000 and  # Reasonable temperature range
            pressure > 10 and pressure < 1e9  # Reasonable pressure range
        )

    def _compute_property(self, field_name: str) -> float:
        """Compute specific thermodynamic property"""

        if field_name == 'Density':
            return self.gas.density_mass
        elif field_name == 'Viscosity':
            return self.gas.viscosity
        elif field_name == 'Conductivity':
            return self.gas.thermal_conductivity
        elif field_name == 'Sound_Speed':
            return self.gas.sound_speed
        elif field_name == 'Cp':
            return self.gas.cp_mass
        elif field_name == 'Cv':
            return self.gas.cv_mass
        elif field_name == 'Mach_Number':
            # Need velocity field for this
            return np.nan
        elif field_name.startswith('D_'):
            species_name = field_name[2:]
            if species_name in self.species_names:
                species_idx = self.gas.species_index(species_name)
                return self.gas.mix_diff_coeffs_mass[species_idx]
        elif field_name.startswith('W_'):
            species_name = field_name[2:]
            if species_name in self.species_names:
                species_idx = self.gas.species_index(species_name)
                return (self.gas.net_production_rates[species_idx] *
                       self.gas.molecular_weights[species_idx])

        return np.nan

    def validate_composition(self, species_dict: Dict[str, float]) -> bool:
        """Validate species composition"""
        try:
            total_mass_fraction = sum(species_dict.values())
            return 0.95 <= total_mass_fraction <= 1.05  # Allow 5% tolerance
        except:
            return False

    def _create_default_logger(self):
        """Create default logger"""
        import logging
        logger = logging.getLogger('CanteraBridge')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

class ThermodynamicCalculator:
    """
    High-level thermodynamic property calculator

    Provides convenient interface for computing thermodynamic properties
    with automatic fallbacks and validation.
    """

    def __init__(self, mechanism_file: Optional[str] = None, logger=None):
        self.logger = logger or self._create_default_logger()

        self.cantera_bridge = None
        if mechanism_file:
            try:
                self.cantera_bridge = CanteraBridge(mechanism_file, logger)
            except Exception as e:
                self.logger.warning(f"Cantera initialization failed: {e}")

    def compute_mach_number(self, data: ProcessedData, velocity_field: str = 'X_Velocity') -> np.ndarray:
        """
        Compute Mach number from velocity and sound speed

        Args:
            data: Processed data
            velocity_field: Name of velocity field

        Returns:
            Array of Mach numbers
        """
        if not data.has_field(velocity_field):
            raise ValidationError(f"Velocity field '{velocity_field}' not found")

        velocity = data.get_field(velocity_field)

        # Try to get sound speed directly
        if data.has_field('Sound_Speed'):
            sound_speed = data.get_field('Sound_Speed')
        elif self.cantera_bridge:
            # Compute using Cantera
            props = self.cantera_bridge.compute_missing_properties(data, ['Sound_Speed'])
            sound_speed = props.get('Sound_Speed')
            if sound_speed is None:
                raise ThermodynamicsError("Could not compute sound speed")
        else:
            raise ThermodynamicsError("Sound speed not available and no Cantera mechanism provided")

        # Compute Mach number
        with np.errstate(divide='ignore', invalid='ignore'):
            mach_number = velocity / sound_speed

        return mach_number

    def estimate_flame_temperature(self, data: ProcessedData,
                                  method: str = 'max_temperature') -> float:
        """
        Estimate flame temperature using various methods

        Args:
            data: Processed data
            method: Estimation method ('max_temperature', 'gradient', 'species')

        Returns:
            Estimated flame temperature
        """
        if not data.has_field('Temperature'):
            raise ValidationError("Temperature field required")

        temp_data = data.get_field('Temperature')

        if method == 'max_temperature':
            return float(np.max(temp_data))

        elif method == 'gradient':
            temp_gradient = np.gradient(temp_data, data.coordinates)
            max_grad_idx = np.argmax(temp_gradient)
            return float(temp_data[max_grad_idx])

        elif method == 'species' and data.has_field('Y_OH'):
            oh_data = data.get_field('Y_OH')
            max_oh_idx = np.argmax(oh_data)
            return float(temp_data[max_oh_idx])

        else:
            raise ValidationError(f"Unknown method or required data missing: {method}")

    def _create_default_logger(self):
        """Create default logger"""
        import logging
        logger = logging.getLogger('ThermodynamicCalculator')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger