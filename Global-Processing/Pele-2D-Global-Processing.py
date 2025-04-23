
import yt, os, multiprocessing, re, time, textwrap, itertools, traceback, sys
from dataclasses import dataclass, field, fields
from typing import Optional
import numpy as np

from scipy.interpolate import RegularGridInterpolator, griddata
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import cKDTree

import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

from general_utilities import *
from pele_utilities import *

yt.set_log_level(0)

########################################################################################################################
# Global Program Setting Variables
########################################################################################################################
version = "1.0.0"

flame_temp = 2500.0  # Temperature for flame contour extraction


########################################################################################################################
# Script Configuration Classes
########################################################################################################################

@dataclass
class DataExtractionConfig:
    Parallelize: bool = False
    Location: Optional[float] = None
    Grid: Optional[np.ndarray] = None


@dataclass
class FieldConfig:
    Name: Optional[str]
    Flag: Optional[bool] = False
    Offset: Optional[float] = 0.0
    Animation: Optional[bool] = False
    Local: Optional[bool] = False
    Collective: Optional[bool] = False


@dataclass
class FlameConfig:
    Position: FieldConfig = field(default_factory=lambda: FieldConfig(Name='Flame Position', Flag=True))
    Velocity: FieldConfig = field(default_factory=lambda: FieldConfig(Name='Flame Velocity', Flag=True))
    RelativeVelocity: FieldConfig = field(default_factory=lambda: FieldConfig(Name='Flame Relative Velocity', Flag=True))
    ThermodynamicState: FieldConfig = field(default_factory=lambda: FieldConfig(Name='Flame Thermodynamic State', Flag=True, Offset=10e-6))
    HeatReleaseRate: FieldConfig = field(default_factory=lambda: FieldConfig(Name='Heat Release Rate', Flag=True))
    FlameThickness: FieldConfig = field(default_factory=lambda: FieldConfig(Name='Flame Thickness', Flag=True))  # fixed typo here too
    SurfaceLength: FieldConfig = field(default_factory=lambda: FieldConfig(Name='Surface Length', Flag=True))
    ReynoldsNumber: FieldConfig = field(default_factory=lambda: FieldConfig(Name='Reynolds Number', Flag=True))


@dataclass
class BurnedGasConfig:
    GasVelocity: FieldConfig = field(default_factory=lambda: FieldConfig(Name='Gas Velocity', Flag=True, Offset=10e-6))
    ThermodynamicState: FieldConfig = field(default_factory=lambda: FieldConfig(Name='Flame Thermodynamic State', Flag=True, Offset=10e-6))


@dataclass
class ShockConfig:
    Position: FieldConfig = field(default_factory=lambda: FieldConfig(Name='Shock Position', Flag=True))
    Velocity: FieldConfig = field(default_factory=lambda: FieldConfig(Name='Shock Velocity', Flag=True))
    PreShockThermodynamicState: FieldConfig = field(default_factory=lambda: FieldConfig(Name='Pre-Shock Thermodynamic State', Flag=True, Offset=10e-6))
    PostShockThermodynamicState: FieldConfig = field(default_factory=lambda: FieldConfig(Name='Post-Shock Thermodynamic State', Flag=True, Offset=-10e-6))


@dataclass
class AnimationConfig:
    LocalWindowSize: FieldConfig = field(default_factory=lambda: FieldConfig(Name='Local Window Size', Offset=1e-3))
    Temperature: FieldConfig = field(default_factory=lambda: FieldConfig(Name='Temperature', Flag=True))
    Pressure: FieldConfig = field(default_factory=lambda: FieldConfig(Name='Pressure', Flag=True))
    GasVelocity: FieldConfig = field(default_factory=lambda: FieldConfig(Name='X Velocity', Flag=True))
    HeatReleaseRate: FieldConfig = field(default_factory=lambda: FieldConfig(Name='Heat Release Rate', Flag=True))
    FlameGeometry: FieldConfig = field(default_factory=lambda: FieldConfig(Name='Flame Geometry', Flag=True))
    FlameThickness: FieldConfig = field(default_factory=lambda: FieldConfig(Name='Flame Thickness', Flag=True))
    Schlieren: FieldConfig = field(default_factory=lambda: FieldConfig(Name='Schlieren', Flag=True))
    StreamLines: FieldConfig = field(default_factory=lambda: FieldConfig(Name='StreamLines', Flag=True))


@dataclass
class ScriptConfig:
    DataExtraction: DataExtractionConfig = field(default_factory=DataExtractionConfig)
    Flame: FlameConfig = field(default_factory=FlameConfig)
    BurnedGas: BurnedGasConfig = field(default_factory=BurnedGasConfig)
    Shock: ShockConfig = field(default_factory=ShockConfig)
    Animation: AnimationConfig = field(default_factory=AnimationConfig)

########################################################################################################################
# Specialized Pele Functions
########################################################################################################################

def flame_geometry(data_dir, output_dir, script_config):
    ############################################################
    # Plotting Helper Functions
    ############################################################
    def plot_contour(raw_contour, sorted_contours, output_dir_path):
        plt.figure(figsize=(8, 6))
        plt.scatter(raw_contour[:, 0], raw_contour[:, 1], color='k', label='Raw Contour')

        # print(len(sorted_contours), sorted_contours)
        if isinstance(sorted_contours, list) and all(isinstance(c, np.ndarray) for c in sorted_contours):
            for contour in sorted_contours:
                contour = np.array(contour)
                plt.plot(contour[:, 0], contour[:, 1], label='Sorted Flame Contour')
        else:
            print(sorted_contours)
            contour = np.array(sorted_contours)
            plt.plot(contour[:, 0], contour[:, 1], label='Sorted Flame Contour')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title(ds.current_time.to_value())
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        filename = ensure_long_path_prefix(os.path.join(output_dir, "Animation-Frames", f"Flame-Geometry-Plt-Files", f"{ds.basename}.png"))
        plt.savefig(filename, format='png')
        plt.close()

    def plot_flame_thickness_and_contour(region_grid, region_temperature, contour_arr, normal_line, interpolator,
                                         output_dir_path):
        # Step 1:
        X, Y = np.meshgrid(np.unique(region_grid[:, 0]), np.unique(region_grid[:, 1]))
        # Step 2: Create a contour plot of the temperature field
        plt.figure(figsize=(8, 6))
        plt.scatter(region_grid[len(region_grid) // 2, 0], region_grid[len(region_grid) // 2, 1], marker='o',
                    color='r', s=100,
                    label=f'Flame Center: ({region_grid[len(region_grid) // 2, 0], region_grid[len(region_grid) // 2, 1]})')
        plt.scatter(X.flatten(), Y.flatten(), c=region_temperature.flatten(), cmap='hot')  # 'c' sets the colors
        plt.scatter(normal_line[:, 0], normal_line[:, 1], c=interpolator(normal_line).flatten(), cmap='hot')
        plt.plot(contour_arr[:, 0], contour_arr[:, 1], label='Sorted Flame Contour')
        plt.xlim(min(X.flatten()), max(X.flatten()))
        plt.ylim(min(Y.flatten()), max(Y.flatten()))
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.colorbar()

        plt.title(f'Flame Normal: {ds.current_time.to_value()}')
        filename = ensure_long_path_prefix(os.path.join(output_dir, "Animation-Frames", f"Flame-Thickness-Plt-Files", f"{ds.basename}.png"))
        plt.savefig(filename, format='png')
        plt.close()

    ############################################################
    # Flame Geometry Processing Functions
    ############################################################
    def sort_by_nearest_neighbors(points, domain_grid):
        buffer = 0.0075 * ds.domain_right_edge.to_value()[1]
        valid_indices = (points[:, 1] >= ds.domain_left_edge.to_value()[1] + buffer) & (
                points[:, 1] <= ds.domain_right_edge.to_value()[1] - buffer)
        points = points[valid_indices]

        # Use cKDTree for more efficient nearest neighbor search
        tree = cKDTree(points)
        origin_idx = np.argmin(np.lexsort((points[:, 0], points[:, 1])))
        order = [origin_idx]
        distance_arr = []
        segments = []
        segment_length = []
        segment_start = 0

        for i in range(1, len(points)):
            distances, indices = tree.query(points[order[i - 1]], k=len(points))
            for neighbor_idx in indices[1:]:  # Skip the first as it's the point itself
                if neighbor_idx not in order:
                    order.append(neighbor_idx)
                    break

            distance_arr.append(np.linalg.norm(points[order[i]] - points[order[i - 1]]))
            if distance_arr[-1] > 50 * (domain_grid[0][1] - domain_grid[0][0]):
                segments.append(points[order][segment_start:i])
                segment_length.append(np.sum(np.linalg.norm(np.diff(segments[-1], axis=0), axis=1)))
                segment_start = i

        # If no segments are appended, set segments equal to points
        if not segments:  # If segments is empty
            segments = [points]
            segment_length = [np.sum(np.linalg.norm(np.diff(segments[-1], axis=0), axis=1))]

        if len(np.concatenate(segments)) < 0.95 * len(points):
            nbrs = NearestNeighbors(n_neighbors=len(points), algorithm='ball_tree').fit(points)
            distances, indices = nbrs.kneighbors(points)
            origin_idx = np.argmin(points[:, 1])
            order = [origin_idx]
            distance_arr = []
            segments = []
            segment_length = []
            segment_start = 0
            for i in range(1, len(points)):
                temp_idx = np.argwhere(indices[:, 0] == order[i - 1])[0][0]
                for neighbor_idx in indices[temp_idx, 1:]:
                    if neighbor_idx not in order:
                        order.append(neighbor_idx)
                        break

                distance_arr.append(np.linalg.norm(points[order[i]] - points[order[i - 1]]))
                if distance_arr[-1] > 50 * (domain_grid[0][1] - domain_grid[0][0]):
                    segments.append(points[order][segment_start:i])
                    segment_length.append(np.sum(np.linalg.norm(np.diff(segments[-1], axis=0), axis=1)))
                    segment_start = i

        return points[order], segments, np.sum(segment_length)

    def manually_aquire_flame_contour(raw_data):
        # Get the maximum refinement level
        max_level = raw_data.index.max_level
        # Initialize containers for the highest level data
        x_coords, y_coords, temperatures = [], [], []
        # Loop through grids and extract data at the highest level
        for grid in raw_data.index.grids:
            if grid.Level == max_level:
                x_coords.append(grid["boxlib", "x"].to_value().flatten())
                y_coords.append(grid["boxlib", "y"].to_value().flatten())
                temperatures.append(grid["Temp"].flatten())

        x_coords = np.concatenate(x_coords)
        y_coords = np.concatenate(y_coords)
        temperatures = np.concatenate(temperatures)

        # Create a triangulation
        triangulation = Triangulation(x_coords, y_coords)

        # Use tricontour to compute the contour line
        contour = plt.tricontour(triangulation, temperatures, levels=[flame_temp])

        # If no contour is found, use grid interpolation
        if not contour.collections:
            print("No contour found at the specified level. Using interpolation...")

            # Create a regular grid for interpolation
            xi = np.linspace(np.min(x_coords), np.max(x_coords), 1e4)
            yi = np.linspace(np.min(y_coords), np.max(y_coords), 1e4)
            xi, yi = np.meshgrid(xi, yi)

            # Interpolate temperatures onto the regular grid
            temperature_grid = griddata((x_coords, y_coords), temperatures, (xi, yi), method='cubic')

            # Create a new triangulation on the regular grid and compute the contour
            triangulation = Triangulation(xi.flatten(), yi.flatten())
            contour = plt.tricontour(triangulation, temperature_grid.flatten(), levels=[flame_temp])

        # Extract the contour line vertices
        paths = contour.collections[0].get_paths()
        contour_points = np.vstack([path.vertices for path in paths])

        return contour_points


    ############################################################
    # Flame Thickness Processing Functions
    ############################################################
    def flame_thickness(contour_arr, center_val, output_dir_path):
        def extract_simulation_grid():
            # Step 1: Extract the flame location from the contour and simulation array
            flame_idx = np.argmin(abs(contour_arr[:, 1] - center_val))
            flame_x, flame_y = contour_arr[flame_idx]

            # Step 2: Collect the max level grids
            max_level = ds.index.max_level
            grids = [grid for grid in ds.index.grids if grid.Level == max_level]

            # Step 3: Pre-allocate lists for subgrid data and filtered grids
            subgrid_x, subgrid_y, subgrid_temperatures = [], [], []
            filtered_grids = []

            # Step 4: Pre-extract the grid data once for efficiency
            grid_data = []
            for temp_grid in grids:
                x = temp_grid["boxlib", "x"].to_value().flatten()
                y = temp_grid["boxlib", "y"].to_value().flatten()
                temp = temp_grid["Temp"].flatten()
                grid_data.append((x, y, temp))

            # Step 5: Filter grids based on mean x difference
            for i, (x, y, temp) in enumerate(grid_data):
                # Calculate the mean x value for the current grid
                current_mean_x = np.mean(x)
                if i < len(grids) - 1:
                    # If the difference in mean x values is too large, skip the current grid
                    if current_mean_x > flame_x + 1e-2:
                        continue

                # If this grid is not skipped, append it to the filtered list
                filtered_grids.append(grids[i])
                # Collect the values from this grid
                subgrid_x.extend(x)
                subgrid_y.extend(y)
                subgrid_temperatures.extend(temp)

            subgrid_x = np.array(subgrid_x)
            subgrid_y = np.array(subgrid_y)
            subgrid_total_temperatures = np.array(subgrid_temperatures)

            # print(np.unique(subgrid_y).shape, np.unique(subgrid_y))
            return subgrid_x, subgrid_y, subgrid_total_temperatures

        def create_subgrid():
            # print(f"Flame X: {flame_x_arr_idx}, Flame Y: {flame_y_arr_idx}")
            # Step 1: Determine the number of indices to the left and right of the flame_x_idx
            left_x_indices = flame_x_arr_idx
            right_x_indices = len(flame_x_arr) - flame_x_arr_idx - 1
            # print(f"Left X Indices: {left_x_indices}, Right X Indices: {right_x_indices}")
            # Determine the smallest number of cells for the x indices
            x_indices = min(left_x_indices, right_x_indices)

            # Step 2: Determine the number of indices to the top and bottom of the flame_y_idx
            top_y_indices = flame_y_arr_idx
            bottom_y_indices = len(flame_y_arr) - flame_y_arr_idx - 1
            # print(f"Top Y Indices: {top_y_indices}, Bottom Y Indices: {bottom_y_indices}")
            # Determine the smallest number of cells for the y indices
            y_indices = min(top_y_indices, bottom_y_indices)

            # Step 3: Determine the subgrid bin size
            if min(x_indices, y_indices) < 11:
                subgrid_bin_size = min(x_indices, y_indices)
            else:
                subgrid_bin_size = 11

            # print(f"Subgrid Bin Size: {subgrid_bin_size}")
            # Step 4: Create subgrid with the appropriate number of indices on either side of flame_x_idx and flame_y_idx
            subgrid_flame_x = flame_x_arr[flame_x_arr_idx - subgrid_bin_size:flame_x_arr_idx + subgrid_bin_size + 1]
            subgrid_flame_y = flame_y_arr[flame_y_arr_idx - subgrid_bin_size:flame_y_arr_idx + subgrid_bin_size + 1]

            # Step 6: Create a grid of temperature values corresponding to the subgrid (subgrid_flame_x, subgrid_flame_y)
            subgrid_temperatures = np.full((len(subgrid_flame_y), len(subgrid_flame_x)), np.nan)

            # Step 6: Create a grid of temperature values corresponding to the subgrid (subgrid_flame_x, subgrid_flame_y)
            # Iterate over the subgrid (x, y) pairs and find the corresponding temperature from the collective data
            for i, y in enumerate(subgrid_flame_y):
                for j, x in enumerate(subgrid_flame_x):
                    # Find the index in the collective data that corresponds to the current (x, y)
                    matching_indices = np.where((subgrid_x == x) & (subgrid_y == y))

                    if len(matching_indices[0]) > 0:
                        # If a match is found, assign the temperature at the (x, y) position
                        try:
                            subgrid_temperatures[i, j] = subgrid_total_temperatures[matching_indices[0][0]]
                        except:
                            subgrid_temperatures[i, j] = np.nan
                            print(f"Error: Unable to assign temperature at ({x}, {y}), "
                                  f"Temperature set to previous value: {subgrid_temperatures[i, j]}")

            region_grid = np.dstack(np.meshgrid(subgrid_flame_x, subgrid_flame_y)).reshape(-1, 2)
            region_temperature = subgrid_temperatures.reshape(np.meshgrid(subgrid_flame_x, subgrid_flame_y)[0].shape)

            break_outer = False
            for i in range(2):
                if i == 0:
                    temp_arr = region_temperature
                else:
                    temp_arr = np.flip(region_temperature, axis=i - 1)

                for j in range(4):
                    temp_grid = np.rot90(temp_arr, k=j)

                    # Compute alignment score (difference between grid and contour points)
                    interpolator = RegularGridInterpolator((np.unique(region_grid[:, 0]), np.unique(region_grid[:, 1])),
                                                           temp_grid, bounds_error=False, fill_value=None)
                    contour_temps = interpolator(region_grid).reshape(
                        np.meshgrid(subgrid_flame_x, subgrid_flame_y)[0].shape)

                    """
                    plt.figure(figsize=(8, 6))
                    plt.imshow(interpolator(region_grid).reshape(np.meshgrid(subgrid_flame_x, subgrid_flame_y)[0].shape),
                               extent=[subgrid_flame_x.min(), subgrid_flame_x.max(), subgrid_flame_y.min(),
                                       subgrid_flame_y.max()],
                               origin='lower', cmap='hot', aspect='auto')
                    plt.colorbar(label='Temperature')
                    plt.show()
                    """

                    if np.all(contour_temps == region_temperature):
                        break_outer = True
                        break  # Break out of the inner loop

                if break_outer:
                    break  # Break out of the outer loop

            return region_grid, region_temperature, interpolator

        def calculate_contour_normal():
            # Step 1: Compute the gradient of the contour points
            dx = np.gradient(contour_arr[:, 0])
            dy = np.gradient(contour_arr[:, 1])

            # Step 2: Compute the normals
            normals = np.zeros_like(contour_arr)
            # Case 1: If the contour is aligned with the x-axis, the normal should be along the y-axis
            for i in range(len(dx)):
                if (dx[i] == 0):  # No change in x-coordinates, thus the normal is along y-axis
                    normals[i, 0] = 0  # Normal along the y-axis (positive direction)
                    normals[i, 1] = 1  # No change in y for normal direction

                # Case 2: If the contour is aligned with the y-axis, the normal should be along the x-axis
                elif (dy[i] == 0):  # No change in y-coordinates, normal should be along x-axis
                    normals[i, 0] = 1  # No change in x for normal direction
                    normals[i, 1] = 0  # Normal along the x-axis (positive direction)

                # General case: Calculate the normal by rotating the tangent 90 degrees
                else:
                    normals[i, 0] = dy[i]  # Rotate by 90 degrees
                    normals[i, 1] = -dx[i]  # Invert the x-component of the tangent

            # Step 3: Normalize the normal vectors
            normals = normals / np.linalg.norm(normals, axis=1)[:, np.newaxis]

            return normals

        def calculate_normal_vector_line(normal_vector):
            # Step 1: Determine the spacing to be used for the normal vector
            dx = np.abs(np.unique(region_grid[:, 0])[1] - np.unique(region_grid[:, 0])[0])
            dy = np.abs(np.unique(region_grid[:, 1])[1] - np.unique(region_grid[:, 1])[0])
            t_step = min(dx, dy) / np.linalg.norm(normal_vector)  # Adjust step size for resolution

            # Center point of the array
            center_point = region_grid[region_grid.shape[0] // 2]

            # Step 2: Determine the bounds for the normal vector
            t_min_x = (np.min(region_grid[:, 0]) - center_point[0]) / normal_vector[0] if normal_vector[
                                                                                              0] != 0 else -np.inf
            t_max_x = (np.max(region_grid[:, 0]) - center_point[0]) / normal_vector[0] if normal_vector[
                                                                                              0] != 0 else np.inf
            t_min_y = (np.min(region_grid[:, 1]) - center_point[1]) / normal_vector[1] if normal_vector[
                                                                                              1] != 0 else -np.inf
            t_max_y = (np.max(region_grid[:, 1]) - center_point[1]) / normal_vector[1] if normal_vector[
                                                                                              1] != 0 else np.inf

            t_start = max(min(t_min_x, t_max_x), min(t_min_y, t_max_y))
            t_end = min(max(t_min_x, t_max_x), max(t_min_y, t_max_y))

            # Step 3: Generate t_range
            t_range = np.arange(t_start, t_end, t_step / 1e2, dtype=np.float32)

            # Step 4: Generate line points along the normal vector
            x_line_points = np.array(center_point[0] + t_range * normal_vector[0], dtype=np.float32)
            y_line_points = np.array(center_point[1] + t_range * normal_vector[1], dtype=np.float32)
            line_points = np.column_stack((x_line_points, y_line_points))

            # Step 5: Filter line points to ensure they remain within bounds
            min_x, max_x = np.min(region_grid[:, 0]), np.max(region_grid[:, 0])
            min_y, max_y = np.min(region_grid[:, 1]), np.max(region_grid[:, 1])
            line_points_filtered = line_points[
                (line_points[:, 0] >= min_x) & (line_points[:, 0] <= max_x) &
                (line_points[:, 1] >= min_y) & (line_points[:, 1] <= max_y)
                ]

            return line_points_filtered

        # Step 1: Extract the flame location from the contour and simulation array
        flame_idx = np.argmin(abs(contour_arr[:, 1] - center_val))
        flame_x, flame_y = contour_arr[flame_idx]

        # Step 2:
        subgrid_x, subgrid_y, subgrid_total_temperatures = extract_simulation_grid()
        # print(subgrid_x.shape, subgrid_y.shape, subgrid_total_temperatures.shape)

        # Step 3: Find the nearest index to the flame contour
        flame_x_idx = np.argmin(np.abs(subgrid_x - flame_x))
        flame_y_idx = np.argmin(np.abs(subgrid_y - flame_y))
        # print(flame_x_idx, flame_y_idx)

        flame_x_arr = subgrid_x[np.abs(subgrid_y - subgrid_y[flame_y_idx]) <= 1e-12]
        flame_y_arr = subgrid_y[np.abs(subgrid_x - subgrid_x[flame_x_idx]) <= 1e-12]
        # print(flame_x_arr.shape, flame_y_arr.shape)

        flame_x_arr_idx = np.argmin(np.abs(flame_x_arr - flame_x))
        flame_y_arr_idx = np.argmin(np.abs(flame_y_arr - flame_y))

        # Step 4:
        region_grid, region_temperature, interpolator = create_subgrid()
        # print(region_grid.shape, region_temperature.shape)

        # Step 5:
        contour_normals = calculate_contour_normal()
        normal_line = calculate_normal_vector_line(contour_normals[flame_idx])
        normal_distances = np.insert(np.cumsum(np.sqrt(np.sum(np.diff(normal_line, axis=0) ** 2, axis=1))), 0, 0)
        normal_line_temperature = interpolator(normal_line)

        # Step 6:
        temp_grad = np.abs(np.gradient(interpolator(normal_line)) / np.gradient(normal_distances))
        try:
            flame_thickness_val = (np.max(interpolator(normal_line)) - np.min(interpolator(normal_line))) / np.max(
                temp_grad)

            temp_plt_dir = ensure_long_path_prefix(
                os.path.join(output_dir_path, "Animation-Frames", "Flame-Thickness-Plt-Files"))
            os.makedirs(temp_plt_dir, exist_ok=True)
            plot_flame_thickness_and_contour(region_grid, region_temperature, contour_arr, normal_line,
                                             interpolator, temp_plt_dir)
        except ValueError as e:
            print(f"Error: Unable to calculate flame thickness: {e}")
            flame_thickness_val = 0

        return flame_thickness_val

    ############################################################
    # Main Function
    ############################################################
    # Step 1: Load the data
    ds = yt.load(data_dir)
    ds.force_periodicity()

    # Step 2:
    tmp_dict = {}

    # Step 3:
    try:
        try:
            contour_verts = manually_aquire_flame_contour(ds)
        except Exception as e:
            contour_verts = ds.all_data().extract_isocontours("Temp", flame_temp)
            print(f"Error: Unable to manually extract flame contour: {e}")

        sorted_points, sorted_segments, contour_length = sort_by_nearest_neighbors(contour_verts, script_config.DataExtraction.Grid)

        if script_config.Flame.FlameThickness.Animation:
            temp_plt_dir = ensure_long_path_prefix(
                os.path.join(output_dir, f"Animation-Frames", f"Surface-Contour-Plt-Files"))

            if os.path.exists(temp_plt_dir) is False:
                os.makedirs(temp_plt_dir, exist_ok=True)

            plot_contour(contour_verts, sorted_segments, temp_plt_dir)

    except Exception as e:
        contour_length = np.nan
        print(f"Error: Unable to extract flame contour: {e}")

    # Compute requested metrics
    if script_config.Flame.SurfaceLength.Flag:
        tmp_dict['Surface Length'] = contour_length / 100
    if script_config.Flame.FlameThickness.Flag:
        if contour_length != 0:
            try:
                tmp_dict['Flame Thickness'] = flame_thickness(sorted_points, script_config.DataExtraction.Location, output_dir) / 100
            except Exception as e:
                tmp_dict['Flame Thickness'] = np.nan
                print(f"Error: Unable to extract flame thickness: {e}")
        else:
            tmp_dict['Flame Thickness'] = np.nan

    return tmp_dict


def reynolds_number():

    return


########################################################################################################################
# Specialized YT Plotting Functions
########################################################################################################################

def schlieren(data_dir, output_dir):
    # Load the data AFTER the field is registered
    ds = yt.load(data_dir)

    # Add gradient fields for the 'density' field
    ds.add_gradient_fields(("boxlib", "density"))

    # Find the maximum location of Y(HO2)
    _, loc = ds.find_max(("boxlib", "Y(HO2)"))
    loc[2] = 0.0

    # Generate the Schlieren image using SlicePlot
    ds.force_periodicity()
    slc = yt.SlicePlot(ds, normal="z", fields=("boxlib", "density_gradient_magnitude"), center=loc)
    slc.set_figure_size(16.0)
    slc.set_width(((1,'code_length'),(ds.domain_right_edge[1].to_value(),'code_length')))
    slc.set_log(("boxlib", "density_gradient_magnitude"), True)
    slc.set_cmap(("boxlib", "density_gradient_magnitude"), "gray")
    slc.annotate_title("Schlieren-like Image")
    slc.save(output_dir)

    return


def streamline(data_dir, output_dir):
    # Load the data AFTER the field is registered
    ds = yt.load(data_dir)

    # Load the data AFTER the field is registered
    ds.force_periodicity()

    # Find the maximum location of Y(HO2)
    _, loc = ds.find_max(("boxlib", "Y(HO2)"))
    loc[2] = 0.0

    # Generate the streamlines using SlicePlot
    slc = yt.SlicePlot(ds, "z", ("boxlib", "x_velocity"), center=loc, width=(ds.domain_right_edge[1] * 10, ds.domain_right_edge[1]))
    slc.annotate_streamlines(("boxlib", "x_velocity"), ("boxlib", "y_velocity"), density=1, broken_streamlines=False)

    # Save the final image as a PNG
    slc.set_figure_size(32.0)
    slc.save(output_dir)

    return


########################################################################################################################
# Parallelization Initialization Functions
########################################################################################################################

def single_file_processing(args):
    ###########################################
    # Internal Functions
    ###########################################
    def load_data(pltFile_dir):
        # Step 1: Load and sort the pelec plot file data
        raw_data = yt.load(pltFile_dir)
        time = raw_data.current_time.to_value()

        # Step 2: Depending on the desired pre-loaded variables
        data, grids = data_extraction(data_dir,
                                      script_config.DataExtraction.Location,
                                      direction='x',
                                      parallelize=script_config.DataExtraction.Parallelize)

        return time, raw_data, data, grids


    ###########################################
    # Main Function
    ###########################################
    # Step 1: Unpack the arguments
    data_dir, output_dir, script_config = args

    # Step 2: Extract the data
    time, raw_data, pre_loaded_data, grids = load_data(data_dir)

    # Step 3: Process the data
    result_dict = {'Time': time}

    # Flame Processing
    if script_config.Flame.Position.Flag:
        result_dict['Flame'] = {}
        # Position
        if script_config.Flame.Position.Flag:
            result_dict['Flame']['Index'], result_dict['Flame']['Position'] = wave_tracking('Flame',
                                                                                            pre_loaded_data=pre_loaded_data)

        # Gas Velocity
        if script_config.Flame.RelativeVelocity.Flag:
            tmp_idx = np.argmin(abs(pre_loaded_data['X'] - (result_dict['Flame']['Position'] + script_config.Flame.RelativeVelocity.Offset)))
            result_dict['Flame']['Gas Velocity'] = pre_loaded_data['X Velocity'][tmp_idx]

        # Thermodynamic State
        if script_config.Flame.ThermodynamicState.Flag:
            result_dict['Flame']['Thermodynamic'] = thermodynamic_state_extractor(pre_loaded_data,
                                                                                  result_dict['Flame']['Position'],
                                                                                  script_config.Flame.ThermodynamicState.Offset)
        # Heat Release Rate
        if script_config.Flame.HeatReleaseRate.Flag:
            result_dict['Flame']['HRR'] = heat_release_rate_extractor(pre_loaded_data,
                                                                      result_dict['Flame']['Position'],
                                                                      script_config.Flame.ThermodynamicState.Offset)
        # Flame Thickness and Surface Length
        if script_config.Flame.FlameThickness.Flag or script_config.Flame.SurfaceLength.Flag:
            tmp_dict = flame_geometry(data_dir, output_dir, script_config)
            # Flame Thickness
            if script_config.Flame.FlameThickness.Flag:
                result_dict['Flame']['Flame Thickness'] = tmp_dict['Flame Thickness']
            # Surface Length
            if script_config.Flame.SurfaceLength.Flag:
                result_dict['Flame']['Surface Length'] = tmp_dict['Flame Thickness']
        # Reynolds Number
        if script_config.Flame.ReynoldsNumber.Flag:
            result_dict['Flame']['Reynolds Number'] = reynolds_number(pre_loaded_data)

    # Burned Gas Processing
    if script_config.Flame.Position.Flag and (script_config.BurnedGas.GasVelocity.Flag or script_config.BurnedGas.ThermodynamicState.Flag):
        result_dict['Burned Gas'] = {}
        if script_config.BurnedGas.GasVelocity.Flag:
            tmp_idx = np.argmin(abs(pre_loaded_data['X'] - (result_dict['Flame']['Position'] - script_config.BurnedGas.GasVelocity.Offset)))
            result_dict['Burned Gas']['Gas Velocity'] = pre_loaded_data['X Velocity'][tmp_idx]
        if script_config.BurnedGas.ThermodynamicState.Flag:
            result_dict['Burned Gas']['Thermodynamic'] = thermodynamic_state_extractor(pre_loaded_data,
                                                                                       result_dict['Flame']['Position'],
                                                                                       script_config.BurnedGas.ThermodynamicState.Offset)

    # Shock Processing
    if script_config.Shock.Position.Flag:
        result_dict['Shock'] = {}
        # Position
        if script_config.Shock.Position.Flag:
            result_dict['Shock']['Index'], result_dict['Shock']['Position'] = wave_tracking('Shock',
                                                                                          pre_loaded_data=pre_loaded_data)

        if script_config.Shock.PreShockThermodynamicState.Flag:
            result_dict['Shock']['PreShockThermodynamicState'] = thermodynamic_state_extractor(pre_loaded_data,
                                                                                              result_dict['Shock']['Position'],
                                                                                              script_config.Shock.PreShockThermodynamicState.Offset)

        if script_config.Shock.PostShockThermodynamicState.Flag:
            result_dict['Shock']['PostShockThermodynamicState'] = thermodynamic_state_extractor(pre_loaded_data,
                                                                                                  result_dict['Shock']['Position'],
                                                                                                  script_config.Shock.PostShockThermodynamicState.Offset)

    # Step 4: Create plot files
    for field_info in fields(script_config.Animation):
        key = field_info.name
        val = getattr(script_config.Animation, key)

        if key in ('FlameGeometry', 'FlameThickness', 'Schlieren', 'StreamLines'):
            continue

        if val.Flag:
            tmp_dir = ensure_long_path_prefix(os.path.join(output_dir, "Animation-Frames", f"{key}-Plt-Files"))
            os.makedirs(tmp_dir, exist_ok=True)

            filename = os.path.join(tmp_dir, f"{raw_data.basename}.png")
            animation_frame_generation(pre_loaded_data['X'], pre_loaded_data[val.Name], key, filename)
            if val.Local:
                tmp_dir = ensure_long_path_prefix(os.path.join(output_dir, "Animation-Frames", f"Local-{key}-Plt-Files"))
                os.makedirs(tmp_dir, exist_ok=True)

                filename = os.path.join(tmp_dir, f"{raw_data.basename}.png")
                animation_frame_generation(pre_loaded_data['X'], pre_loaded_data[val.Name], key, filename,
                                           plot_center=result_dict['Flame']['Position'],
                                           window_size=script_config.Animation.LocalWindowSize.Offset)

    # Step 5: Collective plots
    collected_names = [
        getattr(attr, 'Name')
        for attr in (getattr(script_config.Animation, name) for name in dir(script_config.Animation))
        if getattr(attr, 'Collective', False)
    ]

    if collected_names:
        key = '-'.join(collected_names)
        tmp_dir = ensure_long_path_prefix(os.path.join(output_dir, "Animation-Frames", f"{key}-Plt-Files"))
        os.makedirs(tmp_dir, exist_ok=True)

        collected_arr = []
        for name in collected_names:
            collected_arr.append(pre_loaded_data[name])

        filename = os.path.join(tmp_dir, f"{raw_data.basename}.png")
        animation_frame_generation(
            pre_loaded_data['X'],
            collected_arr,  # Note: val.Name must exist in pre_loaded_data
            collected_names,
            filename
        )

    # Schlieren and Streamlines
    if script_config.Animation.Schlieren.Flag:
        tmp_dir = ensure_long_path_prefix(os.path.join(output_dir, "Animation-Frames", f"Schlieren-Plt-Files"))
        os.makedirs(tmp_dir, exist_ok=True)
        filename = os.path.join(tmp_dir, f"{raw_data.basename}.png")
        # Generate Schlieren
        schlieren(data_dir, filename)

    if script_config.Animation.StreamLines.Flag:
        tmp_dir = ensure_long_path_prefix(os.path.join(output_dir, "Animation-Frames", f"StreamLines-Plt-Files"))
        os.makedirs(tmp_dir, exist_ok=True)
        filename = os.path.join(tmp_dir, f"{raw_data.basename}.png")
        # Generate streamlines
        streamline(data_dir, filename)

    return result_dict


def pelec_processing(data_dirs, output_dir, script_config):
    # Step 1: Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Step 2: Loop over data directories
    result_arr = np.empty(len(data_dirs), dtype=object)
    for i, data_dir in enumerate(data_dirs):
        # Step 3: Process each file
        result_arr[i] = single_file_processing((data_dir, output_dir, script_config))

    # Step 3: Determine the wave velocities
    tmp_time = []
    tmp_flame_position = []
    tmp_shock_position = []
    for i in range(len(result_arr)):
        tmp_time.append(result_arr[i]['Time'])
        if script_config.Flame.Velocity.Flag or script_config.Flame.RelativeVelocity.Flag:
            tmp_flame_position.append(result_arr[i]['Flame']['Position'])
        if script_config.Shock.Velocity.Flag:
            tmp_shock_position.append(result_arr[i]['Shock']['Position'])


    if script_config.Flame.Velocity.Flag or script_config.Flame.RelativeVelocity.Flag:
        tmp_numerator = np.gradient(tmp_flame_position)
        tmp_denominator = np.gradient(tmp_time)
        tmp_flame_velocity = np.divide(tmp_numerator, tmp_denominator)

        for i in range(len(result_arr)):
            if script_config.Flame.Velocity.Flag:
                result_arr[i]['Flame']['Velocity'] = tmp_flame_velocity[i]
            if script_config.Flame.RelativeVelocity.Flag:
                result_arr[i]['Flame']['Velocity'] = tmp_flame_velocity[i] - result_arr[i]['Flame']['Gas Velocity']

    if script_config.Shock.Velocity.Flag:
        tmp_numerator = np.gradient(tmp_shock_position)
        tmp_denominator = np.gradient(tmp_time)

        for i in range(len(result_arr)):
            result_arr[i]['Shock']['Velocity'] = np.divide(tmp_numerator[i], tmp_denominator[i])

    # Step 4: Save the results
    filename = os.path.join(output_dir, f"Processed-Global-Results-V-{version}.txt")
    write_output(result_arr, filename)

    # Step 5: Generate the animations
    parent_dir = ensure_long_path_prefix(os.path.join(output_dir, "Animation-Frames"))
    frame_dirs = [item for item in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, item))]
    for frame_dir in frame_dirs:
        filename = os.path.join(output_dir, f"{frame_dir}-Animation.mp4")
        generate_animation(os.path.join(parent_dir, frame_dir), filename)

    return


########################################################################################################################
# Main Script
########################################################################################################################

def main():

    # Step 1: Define the script parameters
    data_parent_dir = '../2D-Test-Data'

    ddt_plt_dir = os.path.join(data_parent_dir, 'plt332330')
    #ddt_parent_dir = '../../../Domain-Length-284cm/0.09cm-Complete-Domain/Planar-Kernel-Level-6-Part-3'
    #ddt_plt_dir = os.path.join(data_parent_dir, 'ddt_plt')

    # Step 2: Determine the domain parameters
    row_idx = 'Middle'
    domain_info = domain_parameters(ddt_plt_dir, row_idx)

    # Step 2: Set the processing parameters
    script_config = ScriptConfig()
    # Data Extraction Parameters
    script_config.DataExtraction.Parallelize = False
    script_config.DataExtraction.Location = domain_info[1][0][1]  # Set the location for data extraction
    script_config.DataExtraction.Grid = domain_info[-1]
    # Flame Parameters
    script_config.Flame.Position.Flag = True
    script_config.Flame.Velocity.Flag = True
    script_config.Flame.RelativeVelocity.Flag = True
    script_config.Flame.RelativeVelocity.Offset = 10e-6
    script_config.Flame.ThermodynamicState.Flag = True
    #script_config.Flame.ThermodynamicState.Offset = 10e-6
    script_config.Flame.HeatReleaseRate.Flag = True
    script_config.Flame.SurfaceLength.Flag = True
    script_config.Flame.FlameThickness.Flag = True
    script_config.Flame.ReynoldsNumber.Flag = False
    # Burned Gas Parameters
    script_config.BurnedGas.GasVelocity.Flag = True
    script_config.BurnedGas.ThermodynamicState.Flag = True
    #script_config.Flame.ThermodynamicState.Offset = 10e-6
    # Shock Parameters
    script_config.Shock.Position.Flag = True
    script_config.Shock.Velocity.Flag = True
    script_config.Shock.PreShockThermodynamicState.Flag = True
    script_config.Shock.PostShockThermodynamicState.Flag = True
    # Animation Parameters
    script_config.Animation.LocalWindowSize.Offset = 1e-3
    script_config.Animation.Temperature.Flag = True
    script_config.Animation.Temperature.Local = True
    script_config.Animation.Temperature.Collective = True
    script_config.Animation.Pressure.Flag = True
    script_config.Animation.Pressure.Local = True
    script_config.Animation.Pressure.Collective = True
    script_config.Animation.GasVelocity.Flag = True
    script_config.Animation.GasVelocity.Local = True
    script_config.Animation.HeatReleaseRate.Flag = True
    script_config.Animation.HeatReleaseRate.Local = True
    script_config.Animation.FlameGeometry.Flag = True
    script_config.Animation.FlameThickness.Flag = True
    script_config.Animation.Schlieren.Flag = True
    script_config.Animation.StreamLines.Flag = True

    # Step 3: Define the input parameters
    initialize_parameters(
        T=503.15,
        P=10.0 * 100000,
        Phi=1.0,
        Fuel='H2',
        mech='../Chemical-Mechanisms/Li-Dryer-H2-mechanism.yaml',
    ) # ✅ shared state
    # Add species if needed (Only use input_params.species if working with a small chemical mechanism file)
    add_species_vars(input_params.species)

    # Step 4: Collect data directories
    global_data_dir = '../2D-Test-Data'
    data_paths = load_directories(global_data_dir)
    updated_data_paths = sort_files(data_paths)[-2:]
    #updated_data_paths = [updated_data_paths]

    # Step 6: Create the result directories
    os.makedirs(os.path.join(f"Processed-Global-Results-V{version}", f"y-{domain_info[1][0][1]:.3g}cm"),
                exist_ok=True)
    output_dir = os.path.abspath(os.path.join(f"Processed-Global-Results-V{version}", f"y-{domain_info[1][0][1]:.3g}cm"))

    # Step 5: Process the data
    pelec_processing(updated_data_paths, output_dir, script_config)

    return

if __name__ == "__main__":
    main()