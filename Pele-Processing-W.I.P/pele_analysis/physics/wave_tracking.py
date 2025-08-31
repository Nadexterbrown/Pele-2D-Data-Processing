"""
Wave Tracking Algorithms for PELE Analysis

Professional implementation of wave detection algorithms for flames,
shocks, and other wave phenomena in combustion simulations.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum
from scipy import signal
from scipy.interpolate import interp1d

from ..core.data_structures import ProcessedData, WaveData, WaveType
from ..core.exceptions import WaveTrackingError, ValidationError

class WaveDetectionMethod(Enum):
    """Available wave detection methods"""
    THRESHOLD = "threshold"
    GRADIENT = "gradient"
    PEAK_DETECTION = "peak_detection"
    SPECIES_BASED = "species_based"

class WaveTracker:
    """
    Professional wave tracking system for combustion simulations

    Provides robust algorithms for detecting and tracking flames, shocks,
    and other wave phenomena in PELE simulation data.
    """

    def __init__(self, logger=None):
        """
        Initialize wave tracker

        Args:
            logger: Optional logger instance
        """
        self.logger = logger or self._create_default_logger()

        # Default thresholds - can be customized
        self.flame_temp_threshold = 2000.0  # K
        self.pressure_jump_factor = 1.01    # For shock detection
        self.gradient_threshold = 1e6       # For gradient-based detection

        # Detection statistics
        self._detection_stats = {
            'flames_detected': 0,
            'shocks_detected': 0,
            'failed_detections': 0
        }

    def track_flame(self, data: ProcessedData,
                   method: WaveDetectionMethod = WaveDetectionMethod.THRESHOLD,
                   **kwargs) -> WaveData:
        """
        Track flame position in simulation data

        Args:
            data: Processed simulation data
            method: Detection method to use
            **kwargs: Method-specific parameters

        Returns:
            WaveData with flame information

        Raises:
            WaveTrackingError: If detection fails
        """
        if not data.has_field('Temperature'):
            raise WaveTrackingError(
                "Temperature field required for flame tracking",
                wave_type="flame",
                method=method.value
            )

        try:
            if method == WaveDetectionMethod.THRESHOLD:
                result = self._track_flame_threshold(data, **kwargs)
            elif method == WaveDetectionMethod.GRADIENT:
                result = self._track_flame_gradient(data, **kwargs)
            elif method == WaveDetectionMethod.SPECIES_BASED:
                result = self._track_flame_species(data, **kwargs)
            else:
                raise WaveTrackingError(f"Unsupported flame detection method: {method}")

            if result.is_valid():
                self._detection_stats['flames_detected'] += 1
                self.logger.debug(f"Flame detected at position {result.position:.6f}m")
            else:
                self._detection_stats['failed_detections'] += 1

            return result

        except Exception as e:
            self._detection_stats['failed_detections'] += 1
            raise WaveTrackingError(f"Flame tracking failed: {e}", wave_type="flame", method=method.value)

    def track_shock(self, data: ProcessedData,
                   method: WaveDetectionMethod = WaveDetectionMethod.THRESHOLD,
                   **kwargs) -> WaveData:
        """
        Track shock wave position in simulation data

        Args:
            data: Processed simulation data
            method: Detection method to use
            **kwargs: Method-specific parameters

        Returns:
            WaveData with shock information
        """
        if not data.has_field('Pressure'):
            raise WaveTrackingError(
                "Pressure field required for shock tracking",
                wave_type="shock",
                method=method.value
            )

        try:
            if method == WaveDetectionMethod.THRESHOLD:
                result = self._track_shock_threshold(data, **kwargs)
            elif method == WaveDetectionMethod.GRADIENT:
                result = self._track_shock_gradient(data, **kwargs)
            else:
                raise WaveTrackingError(f"Unsupported shock detection method: {method}")

            if result.is_valid():
                self._detection_stats['shocks_detected'] += 1
                self.logger.debug(f"Shock detected at position {result.position:.6f}m")
            else:
                self._detection_stats['failed_detections'] += 1

            return result

        except Exception as e:
            self._detection_stats['failed_detections'] += 1
            raise WaveTrackingError(f"Shock tracking failed: {e}", wave_type="shock", method=method.value)

    def _track_flame_threshold(self, data: ProcessedData,
                              temp_threshold: Optional[float] = None,
                              **kwargs) -> WaveData:
        """Track flame using temperature threshold method"""

        threshold = temp_threshold or self.flame_temp_threshold
        temp_data = data.get_field('Temperature')
        coords = data.coordinates

        # Find points above threshold
        above_threshold = temp_data >= threshold

        if not np.any(above_threshold):
            return WaveData.invalid(WaveType.FLAME, f"No points above {threshold}K")

        # Use last point above threshold as flame position
        threshold_indices = np.where(above_threshold)[0]
        flame_idx = threshold_indices[-1]

        # Refine with species data if available
        flame_idx = self._refine_with_species(data, flame_idx)

        flame_position = coords[flame_idx]
        flame_temp = temp_data[flame_idx]

        wave_data = WaveData(
            wave_type=WaveType.FLAME,
            index=flame_idx,
            position=flame_position,
            detection_method="threshold"
        )

        wave_data.add_property('temperature', flame_temp)
        wave_data.add_property('threshold_used', threshold)

        return wave_data

    def _track_flame_gradient(self, data: ProcessedData, **kwargs) -> WaveData:
        """Track flame using temperature gradient method"""

        temp_data = data.get_field('Temperature')
        coords = data.coordinates

        # Compute temperature gradient
        temp_gradient = np.gradient(temp_data, coords)

        # Find maximum gradient (steepest temperature rise)
        max_grad_idx = np.argmax(temp_gradient)

        # Validate detection
        max_gradient = temp_gradient[max_grad_idx]
        if max_gradient < self.gradient_threshold:
            return WaveData.invalid(WaveType.FLAME, f"Max gradient {max_gradient} below threshold")

        flame_position = coords[max_grad_idx]

        wave_data = WaveData(
            wave_type=WaveType.FLAME,
            index=max_grad_idx,
            position=flame_position,
            detection_method="gradient"
        )

        wave_data.add_property('max_gradient', max_gradient)
        wave_data.add_property('temperature', temp_data[max_grad_idx])

        return wave_data

    def _track_flame_species(self, data: ProcessedData,
                            species: str = "HO2", **kwargs) -> WaveData:
        """Track flame using species concentration"""

        species_field = f'Y_{species}'
        if not data.has_field(species_field):
            return WaveData.invalid(WaveType.FLAME, f"Species {species} not available")

        species_data = data.get_field(species_field)
        coords = data.coordinates

        # Find maximum species concentration
        max_species_idx = np.argmax(species_data)
        max_concentration = species_data[max_species_idx]

        if max_concentration <= 0:
            return WaveData.invalid(WaveType.FLAME, f"No {species} detected")

        flame_position = coords[max_species_idx]

        wave_data = WaveData(
            wave_type=WaveType.FLAME,
            index=max_species_idx,
            position=flame_position,
            detection_method="species_based"
        )

        wave_data.add_property(f'{species}_concentration', max_concentration)

        if data.has_field('Temperature'):
            wave_data.add_property('temperature', data.get_field('Temperature')[max_species_idx])

        return wave_data

    def _track_shock_threshold(self, data: ProcessedData,
                              pressure_factor: Optional[float] = None,
                              **kwargs) -> WaveData:
        """Track shock using pressure jump method"""

        factor = pressure_factor or self.pressure_jump_factor
        pressure_data = data.get_field('Pressure')
        coords = data.coordinates

        # Use end pressure as reference
        reference_pressure = pressure_data[-1]
        threshold_pressure = factor * reference_pressure

        # Find points above threshold
        above_threshold = pressure_data >= threshold_pressure

        if not np.any(above_threshold):
            return WaveData.invalid(WaveType.SHOCK, f"No pressure jump detected (factor {factor})")

        # Use last point above threshold as shock position
        threshold_indices = np.where(above_threshold)[0]
        shock_idx = threshold_indices[-1]

        shock_position = coords[shock_idx]
        shock_pressure = pressure_data[shock_idx]
        pressure_ratio = shock_pressure / reference_pressure

        wave_data = WaveData(
            wave_type=WaveType.SHOCK,
            index=shock_idx,
            position=shock_position,
            detection_method="threshold"
        )

        wave_data.add_property('pressure', shock_pressure)
        wave_data.add_property('pressure_ratio', pressure_ratio)
        wave_data.add_property('reference_pressure', reference_pressure)

        return wave_data

    def _track_shock_gradient(self, data: ProcessedData, **kwargs) -> WaveData:
        """Track shock using pressure gradient method"""

        pressure_data = data.get_field('Pressure')
        coords = data.coordinates

        # Compute pressure gradient
        pressure_gradient = np.gradient(pressure_data, coords)

        # Find maximum positive gradient (pressure rise)
        max_grad_idx = np.argmax(pressure_gradient)
        max_gradient = pressure_gradient[max_grad_idx]

        if max_gradient <= 0:
            return WaveData.invalid(WaveType.SHOCK, "No positive pressure gradient found")

        shock_position = coords[max_grad_idx]

        wave_data = WaveData(
            wave_type=WaveType.SHOCK,
            index=max_grad_idx,
            position=shock_position,
            detection_method="gradient"
        )

        wave_data.add_property('max_pressure_gradient', max_gradient)
        wave_data.add_property('pressure', pressure_data[max_grad_idx])

        return wave_data

    def _refine_with_species(self, data: ProcessedData, initial_idx: int,
                           search_range: int = 10) -> int:
        """Refine flame position using species data"""

        # Try common flame species
        species_candidates = ['HO2', 'OH', 'H']

        for species in species_candidates:
            species_field = f'Y_{species}'
            if data.has_field(species_field):
                species_data = data.get_field(species_field)

                # Search around initial position
                start_idx = max(0, initial_idx - search_range)
                end_idx = min(len(species_data), initial_idx + search_range + 1)

                search_region = species_data[start_idx:end_idx]
                max_species_idx = np.argmax(search_region) + start_idx

                # Use species-based position if close to temperature-based
                if abs(max_species_idx - initial_idx) <= search_range:
                    self.logger.debug(f"Refined flame position using {species} species")
                    return max_species_idx

        return initial_idx

    def compute_wave_velocities(self, positions: List[float],
                               times: List[float]) -> np.ndarray:
        """
        Compute wave velocities from position-time data

        Args:
            positions: Wave positions over time
            times: Corresponding timestamps

        Returns:
            Array of wave velocities

        Raises:
            ValidationError: If input data is invalid
        """
        if len(positions) != len(times):
            raise ValidationError("Positions and times must have same length")

        if len(positions) < 2:
            raise ValidationError("Need at least 2 data points for velocity calculation")

        positions = np.array(positions)
        times = np.array(times)

        # Check for valid data
        valid_mask = np.isfinite(positions) & np.isfinite(times)
        if np.sum(valid_mask) < 2:
            raise ValidationError("Need at least 2 valid data points")

        valid_positions = positions[valid_mask]
        valid_times = times[valid_mask]

        # Compute velocities using gradient
        velocities = np.gradient(valid_positions, valid_times)

        return velocities

    def get_detection_statistics(self) -> Dict[str, int]:
        """Get wave detection statistics"""
        return self._detection_stats.copy()

    def reset_statistics(self):
        """Reset detection statistics"""
        self._detection_stats = {
            'flames_detected': 0,
            'shocks_detected': 0,
            'failed_detections': 0
        }

    def _create_default_logger(self):
        """Create default logger"""
        import logging
        logger = logging.getLogger('WaveTracker')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger