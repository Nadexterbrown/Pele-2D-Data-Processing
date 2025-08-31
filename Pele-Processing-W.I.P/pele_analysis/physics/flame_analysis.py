"""
Comprehensive Flame Analysis for PELE Simulations

Advanced flame analysis including geometry, thickness, consumption rates,
and surface properties for 2D combustion simulations.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from scipy.interpolate import RegularGridInterpolator, griddata
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

from ..core.data_structures import FlameAnalysis, WaveData, WaveType
from ..core.exceptions import FlameAnalysisError, ValidationError
from .wave_tracking import WaveTracker


class FlameGeometryAnalyzer:
    """
    Advanced flame geometry analysis for 2D simulations

    Provides comprehensive flame surface analysis including contour extraction,
    thickness calculation, and surface length measurement.
    """

    def __init__(self, flame_temp_threshold: float = 2500.0, logger=None):
        self.flame_temp_threshold = flame_temp_threshold
        self.logger = logger or self._create_default_logger()

    def analyze_flame_geometry(self, dataset, extraction_location: float,
                               script_config, transport_species: str = 'H2') -> Dict[str, Any]:
        """
        Comprehensive flame geometry analysis

        Args:
            dataset: YT dataset object
            extraction_location: Y-location for analysis
            script_config: Configuration object
            transport_species: Species for consumption rate calculation

        Returns:
            Dictionary with geometry analysis results
        """
        results = {}

        try:
            dataset.force_periodicity()

            # Extract flame contour
            contour_data = self._extract_flame_contour(dataset)
            if contour_data is not None:
                sorted_contour, segments, surface_length = self._process_flame_contour(contour_data)
                results['Surface_Length'] = surface_length / 100  # Convert to meters

                # Flame thickness analysis
                if script_config.Flame.FlameThickness.Flag:
                    thickness = self._calculate_flame_thickness(
                        dataset, sorted_contour, extraction_location)
                    results['Flame_Thickness'] = thickness / 100 if thickness else np.nan

                # Consumption rate analysis
                if script_config.Flame.ConsumptionRate.Flag:
                    consumption_rate, burning_velocity = self._calculate_consumption_rate(
                        dataset, sorted_contour, transport_species)
                    results['Consumption_Rate'] = consumption_rate
                    results['Burning_Velocity'] = burning_velocity

            else:
                results['Surface_Length'] = np.nan
                results['Flame_Thickness'] = np.nan
                results['Consumption_Rate'] = np.nan
                results['Burning_Velocity'] = np.nan

        except Exception as e:
            self.logger.error(f"Flame geometry analysis failed: {e}")
            results.update({
                'Surface_Length': np.nan,
                'Flame_Thickness': np.nan,
                'Consumption_Rate': np.nan,
                'Burning_Velocity': np.nan
            })

        return results

    def _extract_flame_contour(self, dataset) -> Optional[np.ndarray]:
        """Extract flame contour from 2D dataset"""

        try:
            max_level = dataset.index.max_level

            # Collect highest level data
            x_coords, y_coords, temperatures = [], [], []

            for grid in dataset.index.grids:
                if grid.Level == max_level:
                    x_coords.append(grid["boxlib", "x"].to_value().flatten())
                    y_coords.append(grid["boxlib", "y"].to_value().flatten())
                    temperatures.append(grid["Temp"].flatten())

            if not x_coords:
                return None

            x_coords = np.concatenate(x_coords)
            y_coords = np.concatenate(y_coords)
            temperatures = np.concatenate(temperatures)

            # Create triangulation and extract contour
            triangulation = Triangulation(x_coords, y_coords)
            contour = plt.tricontour(triangulation, temperatures,
                                     levels=[self.flame_temp_threshold])
            plt.close()  # Clean up plot

            if not contour.collections:
                self.logger.warning("No flame contour found at specified temperature")
                return None

            # Extract contour vertices
            paths = contour.get_paths()
            if paths:
                contour_points = np.vstack([path.vertices for path in paths])
                return contour_points

            return None

        except Exception as e:
            self.logger.error(f"Contour extraction failed: {e}")
            return None

    def _process_flame_contour(self, contour_points: np.ndarray) -> Tuple[np.ndarray, List, float]:
        """Process and sort flame contour points"""

        # Filter points within domain bounds (avoid boundary artifacts)
        valid_indices = self._filter_boundary_points(contour_points)
        filtered_points = contour_points[valid_indices]

        if len(filtered_points) < 3:
            return contour_points, [contour_points], 0.0

        # Sort points using nearest neighbor approach
        sorted_points, segments, total_length = self._sort_contour_points(filtered_points)

        return sorted_points, segments, total_length

    def _filter_boundary_points(self, points: np.ndarray, buffer_ratio: float = 0.0075) -> np.ndarray:
        """Filter out points too close to domain boundaries"""

        # Assume domain bounds from point distribution
        y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])
        buffer = buffer_ratio * (y_max - y_min)

        valid_mask = (
                (points[:, 1] >= y_min + buffer) &
                (points[:, 1] <= y_max - buffer)
        )

        return np.where(valid_mask)[0]

    def _sort_contour_points(self, points: np.ndarray) -> Tuple[np.ndarray, List, float]:
        """Sort contour points to create continuous curves"""

        if len(points) < 2:
            return points, [points], 0.0

        try:
            # Use cKDTree for efficient nearest neighbor search
            tree = cKDTree(points)

            # Start from leftmost point
            start_idx = np.argmin(points[:, 0])
            ordered_indices = [start_idx]
            remaining_points = set(range(len(points))) - {start_idx}

            current_point = points[start_idx]
            segments = []
            segment_start = 0
            total_length = 0.0

            # Build ordered sequence
            while remaining_points:
                # Find nearest unvisited point
                distances, indices = tree.query(current_point, k=len(points))

                next_idx = None
                for idx in indices[1:]:  # Skip current point
                    if idx in remaining_points:
                        next_idx = idx
                        break

                if next_idx is None:
                    break

                # Check for large gaps (indicates segment break)
                distance = np.linalg.norm(points[next_idx] - current_point)
                if distance > np.percentile(distances[1:len(points) // 4], 95):
                    # End current segment
                    segment = points[ordered_indices[segment_start:]]
                    if len(segment) > 1:
                        segment_length = np.sum(np.linalg.norm(np.diff(segment, axis=0), axis=1))
                        segments.append(segment)
                        total_length += segment_length
                    segment_start = len(ordered_indices)

                ordered_indices.append(next_idx)
                remaining_points.remove(next_idx)
                current_point = points[next_idx]

            # Add final segment
            if segment_start < len(ordered_indices):
                segment = points[ordered_indices[segment_start:]]
                if len(segment) > 1:
                    segment_length = np.sum(np.linalg.norm(np.diff(segment, axis=0), axis=1))
                    segments.append(segment)
                    total_length += segment_length

            ordered_points = points[ordered_indices]

            return ordered_points, segments, total_length

        except Exception as e:
            self.logger.warning(f"Contour sorting failed, using original order: {e}")
            return points, [points], 0.0

    def _calculate_flame_thickness(self, dataset, contour_points: np.ndarray,
                                   center_location: float) -> Optional[float]:
        """Calculate flame thickness using temperature gradient method"""

        try:
            # Find contour point closest to extraction location
            center_y = center_location * 100  # Convert to cm
            center_idx = np.argmin(np.abs(contour_points[:, 1] - center_y))
            flame_center = contour_points[center_idx]

            # Extract local grid around flame
            local_grid, local_temps, interpolator = self._extract_local_grid(
                dataset, flame_center)

            if interpolator is None:
                return None

            # Calculate normal direction
            normal_vector = self._calculate_contour_normal(contour_points, center_idx)

            # Create line normal to flame surface
            normal_line = self._create_normal_line(flame_center, normal_vector, local_grid)

            if len(normal_line) < 10:  # Need sufficient resolution
                return None

            # Interpolate temperature along normal line
            line_temps = interpolator(normal_line)
            line_distances = np.insert(
                np.cumsum(np.linalg.norm(np.diff(normal_line, axis=0), axis=1)), 0, 0)

            # Calculate temperature gradient
            temp_gradient = np.abs(np.gradient(line_temps, line_distances))

            # Flame thickness = temperature range / max gradient
            temp_range = np.max(line_temps) - np.min(line_temps)
            max_gradient = np.max(temp_gradient)

            if max_gradient > 0:
                return temp_range / max_gradient
            else:
                return None

        except Exception as e:
            self.logger.error(f"Flame thickness calculation failed: {e}")
            return None

    def _extract_local_grid(self, dataset, flame_center: np.ndarray,
                            grid_size: int = 11) -> Tuple[np.ndarray, np.ndarray, Optional[RegularGridInterpolator]]:
        """Extract local grid around flame center"""

        try:
            max_level = dataset.index.max_level

            # Get grid spacing
            dx = dataset.index.get_smallest_dx().to_value()

            # Define local grid bounds
            x_center, y_center = flame_center
            half_size = (grid_size // 2) * dx

            x_bounds = [x_center - half_size, x_center + half_size]
            y_bounds = [y_center - half_size, y_center + half_size]

            # Extract data in local region
            region_grids = []
            region_temps = []

            for grid in dataset.index.grids:
                if grid.Level == max_level:
                    grid_x = grid["boxlib", "x"].to_value()
                    grid_y = grid["boxlib", "y"].to_value()

                    # Check if grid overlaps with region
                    if (np.any((grid_x >= x_bounds[0]) & (grid_x <= x_bounds[1])) and
                            np.any((grid_y >= y_bounds[0]) & (grid_y <= y_bounds[1]))):
                        region_grids.append(grid)
                        region_temps.append(grid["Temp"].to_value())

            if not region_grids:
                return None, None, None

            # Create regular grid for interpolation
            x_grid = np.linspace(x_bounds[0], x_bounds[1], grid_size)
            y_grid = np.linspace(y_bounds[0], y_bounds[1], grid_size)

            # Collect all points and temperatures
            all_x, all_y, all_temps = [], [], []
            for grid, temps in zip(region_grids, region_temps):
                x_flat = grid["boxlib", "x"].to_value().flatten()
                y_flat = grid["boxlib", "y"].to_value().flatten()
                temp_flat = temps.flatten()

                all_x.extend(x_flat)
                all_y.extend(y_flat)
                all_temps.extend(temp_flat)

            # Interpolate to regular grid
            points = np.column_stack([all_x, all_y])
            X, Y = np.meshgrid(x_grid, y_grid)
            grid_points = np.dstack([X, Y]).reshape(-1, 2)

            interpolated_temps = griddata(points, all_temps, grid_points, method='cubic')
            temp_grid = interpolated_temps.reshape(X.shape)

            # Create interpolator
            interpolator = RegularGridInterpolator((y_grid, x_grid), temp_grid,
                                                   bounds_error=False, fill_value=np.nan)

            return grid_points.reshape(X.shape + (2,)), temp_grid, interpolator

        except Exception as e:
            self.logger.error(f"Local grid extraction failed: {e}")
            return None, None, None

    def _calculate_contour_normal(self, contour_points: np.ndarray,
                                  center_idx: int) -> np.ndarray:
        """Calculate normal vector at contour point"""

        # Use finite differences for tangent calculation
        if center_idx == 0:
            tangent = contour_points[1] - contour_points[0]
        elif center_idx == len(contour_points) - 1:
            tangent = contour_points[-1] - contour_points[-2]
        else:
            tangent = contour_points[center_idx + 1] - contour_points[center_idx - 1]

        # Normal is perpendicular to tangent
        normal = np.array([-tangent[1], tangent[0]])

        # Normalize
        normal_magnitude = np.linalg.norm(normal)
        if normal_magnitude > 0:
            normal /= normal_magnitude

        return normal

    def _create_normal_line(self, center_point: np.ndarray, normal_vector: np.ndarray,
                            local_grid: np.ndarray) -> np.ndarray:
        """Create line points along normal direction"""

        # Determine line extent based on local grid
        grid_bounds = np.array([
            [np.min(local_grid[:, :, 0]), np.min(local_grid[:, :, 1])],
            [np.max(local_grid[:, :, 0]), np.max(local_grid[:, :, 1])]
        ])

        # Calculate intersection with grid bounds
        grid_size = np.linalg.norm(grid_bounds[1] - grid_bounds[0])
        line_length = grid_size * 0.8  # Use 80% of grid diagonal

        # Create line points
        n_points = 50
        t_values = np.linspace(-line_length / 2, line_length / 2, n_points)

        line_points = center_point + np.outer(t_values, normal_vector)

        # Filter points within grid bounds
        valid_mask = (
                (line_points[:, 0] >= grid_bounds[0, 0]) &
                (line_points[:, 0] <= grid_bounds[1, 0]) &
                (line_points[:, 1] >= grid_bounds[0, 1]) &
                (line_points[:, 1] <= grid_bounds[1, 1])
        )

        return line_points[valid_mask]

    def _calculate_consumption_rate(self, dataset, contour_points: np.ndarray,
                                    transport_species: str) -> Tuple[float, float]:
        """Calculate species consumption rate and burning velocity"""

        try:
            from ..core.units import convert_to_si

            # Define bounding box around flame
            x_min = np.min(contour_points[:, 0]) - 1e-3  # 1mm buffer
            x_max = np.max(contour_points[:, 0]) + 1e-3
            y_min = np.min(contour_points[:, 1])
            y_max = np.max(contour_points[:, 1])

            # Extract data in bounding box
            max_level = dataset.index.max_level
            left_edge = dataset.arr([x_min, y_min, 0.0], "cm")
            right_edge = dataset.arr([x_max, y_max, dataset.domain_right_edge[2].to_value()], "cm")

            dims = ((right_edge - left_edge) / dataset.index.get_smallest_dx()).to('').astype("int")
            dims[2] = 1

            cg = dataset.covering_grid(
                level=max_level,
                left_edge=left_edge,
                dims=dims,
                fields=[('boxlib', 'x'), ('boxlib', 'y'),
                        ('boxlib', f'rho_{transport_species}'),
                        ('boxlib', f'rho_omega_{transport_species}')]
            )

            # Calculate consumption rate by integrating production rate
            dx = convert_to_si(cg.dds[0].to_value(), 'cm')
            dy = convert_to_si(cg.dds[1].to_value(), 'cm')

            production_rate_data = cg['boxlib', f'rho_omega_{transport_species}'].to_value()
            production_rate_si = convert_to_si(production_rate_data, 'g/cm³/s')

            # Integrate over volume
            total_consumption = np.sum(production_rate_si) * dx * dy

            # Get reactant density for burning velocity calculation
            reactant_density_data = cg["boxlib", f"rho_{transport_species}"].to_value()
            reactant_density_si = convert_to_si(reactant_density_data[-1, 0], 'g/cm³')

            # Calculate burning velocity
            flame_surface_area = convert_to_si(dataset.domain_right_edge[1].to_value(), 'cm')
            burning_velocity = total_consumption / reactant_density_si / flame_surface_area

            return total_consumption, burning_velocity

        except Exception as e:
            self.logger.error(f"Consumption rate calculation failed: {e}")
            return np.nan, np.nan

    def _create_default_logger(self):
        """Create default logger"""
        import logging
        logger = logging.getLogger('FlameGeometryAnalyzer')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger


class FlameAnalyzer:
    """
    Comprehensive flame analysis coordinator

    Coordinates various flame analysis components to provide complete
    flame characterization from PELE simulation data.
    """

    def __init__(self, logger=None):
        self.logger = logger or self._create_default_logger()
        self.wave_tracker = WaveTracker(logger)
        self.geometry_analyzer = FlameGeometryAnalyzer(logger=logger)

    def analyze_flame(self, data: ProcessedData, dataset=None,
                      config=None, **kwargs) -> FlameAnalysis:
        """
        Complete flame analysis

        Args:
            data: Processed 1D simulation data
            dataset: Original yt dataset (for 2D analysis)
            config: Configuration object
            **kwargs: Additional analysis parameters

        Returns:
            FlameAnalysis object with complete results
        """

        # Track flame position in 1D data
        flame_wave = self.wave_tracker.track_flame(data, **kwargs)

        # Initialize flame analysis
        flame_analysis = FlameAnalysis(position_data=flame_wave)

        if not flame_wave.is_valid():
            return flame_analysis

        # Add thermodynamic state
        if data.has_field('Temperature') and data.has_field('Pressure'):
            temp = data.get_field('Temperature')[flame_wave.index]
            pressure = data.get_field('Pressure')[flame_wave.index]

            flame_analysis.add_thermodynamic_property('temperature', temp)
            flame_analysis.add_thermodynamic_property('pressure', pressure)

            if data.has_field('Density'):
                density = data.get_field('Density')[flame_wave.index]
                flame_analysis.add_thermodynamic_property('density', density)

        # Add heat release rate
        if data.has_field('Heat_Release_Rate'):
            flame_analysis.heat_release_rate = data.get_field('Heat_Release_Rate')[flame_wave.index]

        # 2D geometry analysis if dataset provided
        if dataset is not None and config is not None:
            try:
                geometry_results = self.geometry_analyzer.analyze_flame_geometry(
                    dataset, data.extraction_params.location if data.extraction_params else 0.05,
                    config, **kwargs)

                flame_analysis.thickness = geometry_results.get('Flame_Thickness')
                flame_analysis.surface_length = geometry_results.get('Surface_Length')
                flame_analysis.consumption_rate = geometry_results.get('Consumption_Rate')
                flame_analysis.burning_velocity = geometry_results.get('Burning_Velocity')

            except Exception as e:
                self.logger.error(f"2D geometry analysis failed: {e}")

        return flame_analysis

    def _create_default_logger(self):
        """Create default logger"""
        import logging
        logger = logging.getLogger('FlameAnalyzer')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger