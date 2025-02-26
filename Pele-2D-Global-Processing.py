import os, yt, itertools, multiprocessing, textwrap, re, sys
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from sklearn.neighbors import NearestNeighbors
from sdtoolbox.thermo import soundspeed_fr
from threading import Timer
import matplotlib.animation as animation
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cantera as ct
import numpy as np

yt.set_log_level(0)

n_proc = 24
polyfit_bin_size = 51
plotting_bnds_bin = 3
flame_thickness_bin_size = 11


class MyClass():
    def __init__(self):
        # Initialize default parameters
        self.T = None  # Temperature (Kelvin)
        self.P = None  # Pressure (Pa)
        self.Phi = None  # Equivalence ratio
        self.Fuel = None  # Fuel type
        self.mech = None  # Mechanism file
        self.species = None  # Species list
        self.oxygenAmount = None  # Oxygen amount
        self.nitrogenAmount = None  # Nitrogen amount
        self.X = {}  # Composition dictionary

    def update_composition(self):
        """Update the oxygen and nitrogen amounts based on the fuel type."""
        if self.Fuel == "H2":
            self.oxygenAmount = 0.5
        elif self.Fuel == "C2H6":
            self.oxygenAmount = 3.5
        elif self.Fuel == "C4H10":
            self.oxygenAmount = 6.5
        else:
            raise ValueError(f"Unknown fuel type: {self.Fuel}")

        # Update the composition dictionary
        self.X = {
            self.Fuel: self.Phi,
            'O2': self.oxygenAmount,
            'N2': self.nitrogenAmount
        }

    def load_mechanism_species(self):
        """Load species from the mechanism file and store them in the species attribute."""
        if not self.mech:
            raise ValueError("Mechanism file not specified.")
        try:
            gas = ct.Solution(self.mech)
            self.species = gas.species_names
            del gas  # Release Cantera object
        except Exception as e:
            raise RuntimeError(f"Failed to load mechanism file: {e}")


# Global instance of MyClass
input_params = MyClass()


def initialize_parameters(T, P, Phi, Fuel, mech, nitrogenAmount=0):
    """Initialize the parameters with provided values."""
    input_params.T = T
    input_params.P = P
    input_params.Phi = Phi
    input_params.Fuel = Fuel
    input_params.mech = mech
    input_params.nitrogenAmount = nitrogenAmount
    input_params.update_composition()  # Update oxygen amount and composition
    input_params.load_mechanism_species()  # Load species from the mechanism file


# Script Functions
def parallel_processing_function(iter_arr, const_list, predicate, nProcs):
    """

    :param iter_arr:
    :param const_list:
    :param predicate:
    :param nProcs:
    :return:
    """
    # Perform the multiprocessing
    with multiprocessing.Pool(
            processes=nProcs, initargs=()
    ) as pool:
        y = pool.map(predicate,
                     zip(iter_arr,
                         itertools.repeat(const_list),
                         itertools.repeat(input_params)))
    return y


def ensure_long_path_prefix(path):
    """
    Ensure that the path uses the long path prefix if necessary.

    Parameters:
    - path: str, the original file path.

    Returns:
    - str, the path with the long path prefix.
    """
    if path.startswith(r"\\"):
        return r"\\?\UNC" + path[1:]  # UNC path prefix
    else:
        return r"\\?\\" + path  # Regular path prefix


def sort_files(file_list):
    """

    Args:
        file_list: A list of plot files to be sorted

    Returns:
        sorted_list: A list of sorted file paths
    """

    def extract_number(file_name):
        # This regex assumes the number is at the end of the file name
        match = re.search(r'plt(\d+)$', file_name)
        return int(match.group(1)) if match else float('inf')

    # Sort the list based on the extracted number
    sorted_list = sorted(file_list, key=extract_number)
    return sorted_list


def smoothing_function(collective_results, master_key, slave_key):
    def polynomial_fit_over_array(x, y, bin_size=51, degree=1):
        """
        Fits a polynomial of the specified degree to the data within bins
        centered around each point in the input array, and evaluates the
        polynomial at each point.

        Parameters:
            x (list or numpy array): The x-coordinates of the data points.
            y (list or numpy array): The y-coordinates of the data points.
            bin_size (int): The size of the bin around each point.
            degree (int): The degree of the polynomial to fit.

        Returns:
            x_fit (numpy array): The x-coordinates of the data points matched to the output.
            values (numpy array): The evaluated values of the polynomial at each point.
        """
        values = []
        derivative = []
        x_fit = []

        half_bin = bin_size // 2

        for i, point in enumerate(x):
            if i >= half_bin and i <= (len(x) - half_bin):
                # Find the indices within the bin
                center_index = np.argmin(np.abs(x - point))
                start_index = int(i - half_bin)
                end_index = int(i + half_bin)

                # Fit a polynomial to the data within the bin
                x_bin = np.array(x[start_index:end_index], dtype=float)
                y_bin = np.array(y[start_index:end_index], dtype=float)

                coefficients = np.polyfit(x_bin, y_bin, degree)

                # Evaluate the polynomial at the given point
                value = np.polyval(coefficients, point)
                values.append(value)
                x_fit.append(point)

                # Calculate the derivative of the polynomial at the given point
                poly_derivative = np.polyder(coefficients)
                derivative_value = np.polyval(poly_derivative, point)
                derivative.append(derivative_value)

        return np.array(x_fit), np.array(values), np.array(derivative)

    """

    """
    # Step 1: Create the smooth sub-sub-dictonary within the results directory if one does not exist
    if not collective_results[master_key].get('Smooth', False):
        collective_results[master_key]["Smooth"] = {}

    # Step 2: Create a new sub-sub-sub dict pertaining to the desired variable
    collective_results[master_key]["Smooth"][slave_key] = {}

    # Step 3:
    if slave_key == 'Thermodynamic State':
        temp_var_arr = np.empty((len(collective_results[master_key][slave_key]), 4), dtype=object)
        for i in range(len(collective_results[master_key][slave_key])):
            temp_var_arr[i, 0] = collective_results[master_key][slave_key][i][0]
            temp_var_arr[i, 1] = collective_results[master_key][slave_key][i][1]
            temp_var_arr[i, 2] = collective_results[master_key][slave_key][i][2]
            temp_var_arr[i, 3] = collective_results[master_key][slave_key][i][3]

        collective_results[master_key]["Smooth"][slave_key] = np.empty(4, dtype=object)
        for i in range(len(collective_results[master_key]["Smooth"][slave_key])):
            [temp_time, temp_vec, _] = polynomial_fit_over_array(np.array(collective_results['Time']['Value']),
                                                                 np.array(temp_var_arr[:, i]),
                                                                 bin_size=polyfit_bin_size)
            collective_results[master_key]["Smooth"][slave_key][i] = temp_vec
    else:
        [temp_time, temp_vec, _] = polynomial_fit_over_array(np.array(collective_results['Time']['Value']),
                                                             np.array(collective_results[master_key][slave_key]),
                                                             bin_size=polyfit_bin_size)

        collective_results[master_key]["Smooth"][slave_key] = temp_vec

    if not collective_results['Time'].get('Smooth', False):
        collective_results['Time']["Smooth"] = {}
        collective_results['Time']['Smooth']['Value'] = temp_time

    return collective_results


def mechanism_species(mechanism_file):
    """

    Args:
        mechanism_file:

    Returns:

    """
    # Step 1: Create cantera gas object from the provided mechanism file
    gas = ct.Solution(mechanism_file)
    speces_names = gas.species_names
    del gas
    return speces_names


def domain_size_parameters(directory_path, desired_y_location):
    """

    :param directory_path:
    :param desired_y_location:
    :return:
    """
    # Step 2: Load the data for physical size extraction
    ds = yt.load(directory_path)
    max_level = ds.index.max_level

    # Create sudo-grid at the maximum level present
    data = ds.covering_grid(level=max_level,
                            left_edge=[0.0, 0.0, 0.0],
                            dims=ds.domain_dimensions * [2 ** max_level, 2 ** max_level, 1],
                            # And any fields to preload (this is optional!)
                            # fields=desired_varables
                            )

    grid_arr = np.empty(2, dtype=object)
    grid_arr[0] = data["boxlib", "x"][:, 0].to_value()
    grid_arr[1] = data["boxlib", "y"][0, :].to_value()

    # Step 3:
    if isinstance(desired_y_location, str) is True:
        if desired_y_location == "Bottom":
            y_slice_index = 0
            y_slice_loc = data.LeftEdge[1].to_value()
        elif desired_y_location == "Top":
            y_slice_index = data.ActiveDimensions[1] - 1
            y_slice_loc = data.RightEdge[1].to_value()
        elif desired_y_location == "DDT":
            y_slice_index = np.unravel_index(np.argmax(data["boxlib", 'density'].to_value(), axis=None),
                                             data["boxlib", 'density'].to_value().shape)
            y_slice_loc = data["boxlib", str('y')][0][y_slice_index].to_value()[0]
        else:
            y_slice_index = int((data.ActiveDimensions[1] / 2) - 1)
            y_slice_loc = data["boxlib", str('y')][0][y_slice_index].to_value()[0]
    else:
        y_slice_index = np.argwhere(data["boxlib", str('y')][0][:].to_value() <= desired_y_location)[-1][0]
        y_slice_loc = data["boxlib", str('y')][0][y_slice_index].to_value()[0]

    return (np.array([[0, int(y_slice_index)], [int(data.ActiveDimensions[0]), int(y_slice_index)]]),
            np.array([[data.LeftEdge[0].to_value(), y_slice_loc], [data.RightEdge[0].to_value(), y_slice_loc]]),
            grid_arr)


def wave_tracking_function(raw_data, sort_arr, tracking_str, wave_type):
    """

    Args:
        raw_data:
        tracking_str:
        wave_type:

    Returns:

    """
    # Step 1: Load the desired marker and x positions
    temp_data = raw_data["boxlib", str(tracking_str)][sort_arr].to_value()
    temp_x_pos = raw_data["boxlib", str('x')][sort_arr].to_value()
    # Step 2:
    if wave_type == "Flame":
        wave_index = np.argwhere(temp_data >= 2000)[-1][0]
    elif wave_type == "Maximum Pressure":
        temp_index = temp_data
        wave_index = np.argwhere(temp_index == np.max(temp_index))[-1][0]
    elif wave_type == "Leading Shock":
        # From the farthest right point in the domain, determine the leading pressure wave location by 1% increase from this value
        pressure_baseline = temp_data[-1]
        try:
            wave_index = np.argwhere(temp_data >= 1.01 * pressure_baseline)[-1][0]
        except:
            wave_index = 0
    # Step 3:
    plt_check = False
    if plt_check:
        plt.figure(figsize=(8, 6))
        plt.plot(temp_x_pos, temp_data, linestyle='-', color='k')
        plt.plot(temp_x_pos[wave_index], temp_data[wave_index], marker='.', linestyle='-', color='r')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f"{tracking_str}")
        plt.show()

    return wave_index, temp_x_pos[wave_index] / 100


def thermodynamic_state_function(raw_data, sort_arr, wave_idx, wave_type, input_params):
    # Step 1: Collect the position of the wave
    wave_pos = raw_data["boxlib", str('x')][sort_arr].to_value()[wave_idx]

    # Step 2:
    if wave_type == "Flame":
        probe_index = np.argwhere(raw_data["boxlib", str('x')][sort_arr].to_value() >= wave_pos + (-0.001))[0][0]
    elif wave_type == "Maximum Pressure":
        try:
            probe_index = np.argwhere(raw_data["boxlib", str('x')][sort_arr].to_value() >= wave_pos)[0][0]
        except:
            probe_index = len(raw_data["boxlib", str('x')][sort_arr]) - 1
    elif wave_type == "Pre-Shock" or wave_type == "Post-Shock":
        if wave_type == "Pre-Shock":
            try:
                probe_index = np.argwhere(raw_data["boxlib", str('x')][sort_arr].to_value() >= wave_pos + (1 / 10))[0][
                    0]
            except:
                probe_index = len(raw_data["boxlib", str('x')][sort_arr]) - 1

        elif wave_type == "Post-Shock":
            try:
                probe_index = np.argwhere(raw_data["boxlib", str('x')][sort_arr].to_value() >= wave_pos - (1 / 10))[0][
                    0]
            except:
                probe_index = len(raw_data["boxlib", str('x')][sort_arr]) - 1

    # Step 3: Extract the state variables from the raw data
    temperature = raw_data["boxlib", str('Temp')][sort_arr][probe_index].to_value()
    pressure = raw_data["boxlib", str('pressure')][sort_arr][probe_index].to_value()

    species_comp = {}
    for i in range(len(input_params.species)):
        species_comp.update({str(input_params.species[i]):
                                 raw_data["boxlib", str("Y(" + input_params.species[i] + ")")][sort_arr][
                                     probe_index].to_value()})

    # Step 3: Create a Cantera object to extract soundspeed
    gas_obj = ct.Solution(input_params.mech)
    gas_obj.TPY = (temperature,
                   pressure / 10,
                   species_comp)

    # Step 4:
    result_array = np.zeros(4, dtype=float)
    result_array[0] = gas_obj.T
    result_array[1] = gas_obj.P
    result_array[2] = gas_obj.density_mass
    result_array[3] = soundspeed_fr(gas_obj)
    del gas_obj

    return result_array.tolist()


def heat_release_rate_function(raw_data, sort_arr, input_params):
    # Step 1: Collect the partial molar enthalpies and net production rates
    net_production_rates = np.empty(len(raw_data["boxlib", str('x')][sort_arr].to_value()), dtype=object)
    partial_molar_enthalpies = np.empty(len(raw_data["boxlib", str('x')][sort_arr].to_value()), dtype=object)
    for i in range(len(raw_data["boxlib", str('x')][sort_arr].to_value())):
        species_comp = {}
        temp_obj = ct.Solution(input_params.mech)
        for j in range(len(temp_obj.species_names)):
            species_comp.update({str(temp_obj.species_names[j]):
                                     raw_data["boxlib", str("Y(" + temp_obj.species_names[j] + ")")][sort_arr][
                                         i].to_value()})

        temp_obj.TPY = (raw_data["boxlib", str("Temp")][sort_arr][i].to_value(),
                        raw_data["boxlib", str("pressure")][sort_arr][i].to_value() * 10,
                        species_comp)

        net_production_rates[i] = temp_obj.net_production_rates
        partial_molar_enthalpies[i] = temp_obj.partial_molar_enthalpies

        del temp_obj

    # Step 3:
    heat_release_rate = np.zeros(len(raw_data["boxlib", str('x')][sort_arr].to_value()))
    for i in range(len(raw_data["boxlib", str('x')][sort_arr].to_value())):
        integral = 0
        for j in range(len(input_params.species)):
            integral = integral + net_production_rates[i][j] * partial_molar_enthalpies[i][j]

        heat_release_rate[i] = -1.0 * integral

    return heat_release_rate, np.max(heat_release_rate)


def flame_geometry_function(raw_data, center_loc, grid, output_dir_path=None, contour_check=False,
                            thickness_check=False):
    def sort_by_nearest_neighbors(points, plt_check=False):
        # Convert points to numpy array if it's a list of lists
        points = np.array(points)

        # Create NearestNeighbors instance
        nbrs = NearestNeighbors(n_neighbors=len(points), algorithm='ball_tree').fit(points)
        distances, indices = nbrs.kneighbors(points)

        # Construct the sorted order
        # distance_metric = lambda x: (x[0] - 0) ** 2 + (x[1] - 0) ** 2
        # origin_point = min(points, key=distance_metric)
        # origin_idx = np.argwhere(points == origin_point)[0][0]
        origin_idx = np.argmin(points[:, 1])

        # Create viable points array
        order = [origin_idx]
        distance_arr = []

        for i in range(1, len(points)):
            neighbors = []
            temp_idx = np.argwhere(indices[:, 0] == order[i - 1])[0][0]
            for neighbor_idx in indices[temp_idx, 1::]:
                if neighbor_idx not in order and neighbor_idx != indices[temp_idx, 0]:
                    neighbors.append(neighbor_idx)
                    # Mark the neighbor as visited
                    order.append(neighbor_idx)
                    distance_arr.append(distances[temp_idx, np.argwhere(indices[temp_idx, :] == neighbor_idx)])
                    # Stop if we've found enough neighbors for the given point
                    break
        # Check surface length from distances for comparison
        surface_approx = np.sum(distance_arr)
        # print('Approximate Surface Length =', surface_approx, 'cm')
        return points[order]

    def flame_thickness(raw_data, contour_arr, center_loc, grid):
        def compute_contour_normal(points):
            # Create a placeholder array for the tangent
            dx = np.gradient(points[:, 0])
            dy = np.gradient(points[:, 1])
            # Compute normal vectors
            normals = np.zeros_like(points)
            normals[:, 0] = dy  # dy
            normals[:, 1] = -dx  # dx

            normal_vect = normals / np.linalg.norm(normals)
            return normal_vect

        def closest_point_from_poly(points, line):

            # Step 1: Generate points along the line defined by the unit vector
            # Define a range for the line, e.g., from -10 to 10 in the direction of the unit vector
            t_range = np.linspace(-10, 10, 10000)
            line_points = np.column_stack(
                (grid[0][flame_x_idx] + t_range * line[0], grid[1][flame_y_idx] + t_range * line[1]))

            within_bounds_mask = (
                    (line_points[:, 0] >= min(points[:, 0])) & (line_points[:, 0] <= max(points[:, 0])) &
                    (line_points[:, 1] >= min(points[:, 1])) & (line_points[:, 1] <= max(points[:, 1]))
            )
            line_points = line_points[within_bounds_mask]

            # Step 3: Determine the nearest point to the line generated by the unit vector
            min_distance_indices = []

            for line_point in line_points:
                line_x, line_y = line_point
                distances = []

                for point in points:
                    x, y = point
                    # Calculate perpendicular distance to the line point
                    distance = np.sqrt((line_x - x) ** 2 + (line_y - y) ** 2)
                    distances.append(distance)

                # Find the index of the minimum distance for this point
                min_distance_index = np.argmin(distances)
                min_distance_indices.append(min_distance_index)

            min_distance_indices = np.unique(min_distance_indices)
            closest_points = [points[idx] for idx in min_distance_indices]

            return min_distance_indices, np.array(closest_points)

        # Step 1:
        normal_vect = compute_contour_normal(contour_arr)
        # Step 2:
        flame_idx = np.argmin(abs(contour_arr[:, 1] - center_loc))
        flame_norm = normal_vect[flame_idx]

        # Find the location of the flame
        flame_x_idx = np.argmin(abs(grid[0] - contour_arr[flame_idx, 0]))
        flame_y_idx = np.argmin(abs(grid[1] - contour_arr[flame_idx, 1]))

        # Create a sudo-grid region around the center flame point
        region = raw_data.box(np.array([grid[0][flame_x_idx - (flame_thickness_bin_size // 2) - 1][0],
                                        grid[1][flame_y_idx - (flame_thickness_bin_size // 2) - 1][0], 0.0]),
                              np.array([grid[0][flame_x_idx + (flame_thickness_bin_size // 2)][0],
                                        grid[1][flame_y_idx + (flame_thickness_bin_size // 2)][0], 1.0]))

        region_mesh_x, region_mesh_y = np.meshgrid(np.unique(region["x"].to_value()), np.unique(region["y"].to_value()))

        region_grid = np.dstack((region["x"].to_value(), region["y"].to_value()))[0]
        # Determine the closes points in the region grid to the line created by the norm
        nearest_norm_idx, nearest_norm_points = closest_point_from_poly(region_grid, flame_norm)

        # Collect the temperature points nearest the flame surface norm
        region_data = region['Temp'][nearest_norm_idx].to_value()

        # Step 2: Compute the gradient of temperature with respect to position
        # temperature_gradient = abs(np.gradient(region_data) / np.gradient(np.sqrt(temp_grad_x[len()]**2 + temp_grad_y[]**2)))
        # temperature_gradient_idx = np.argwhere(temperature_gradient == np.max(temperature_gradient))[0][0]

        # Step 3:
        dx = np.diff(nearest_norm_points[:, 0] / 100)  # cm to m
        dy = np.diff(nearest_norm_points[:, 1] / 100)  # cm to m

        dx[dx == 0] = np.nan
        dy[dy == 0] = np.nan

        temp_grad_x = np.diff(region_data) / dx
        temp_grad_y = np.diff(region_data) / dy

        temp_grad_x = np.nan_to_num(temp_grad_x, nan=0)
        temp_grad_y = np.nan_to_num(temp_grad_y, nan=0)

        flame_thickness_val = (np.max(region_data) - np.min(region_data)) / np.max(
            np.sqrt(temp_grad_x ** 2 + temp_grad_y ** 2))

        plt_check = False
        if plt_check:
            plt.figure(figsize=(8, 6))
            plt.plot(np.array(contour_arr[:, 0], dtype=float), np.array(contour_arr[:, 1], dtype=float), color='r')
            plt.scatter(region_grid[:, 0], region_grid[:, 1], marker='o', color='k')
            plt.scatter(grid[0][flame_x_idx][0], grid[1][flame_y_idx][0], marker='o', color='r')
            plt.scatter(nearest_norm_points[:, 0], nearest_norm_points[:, 1], marker='o', color='b')
            # plt.scatter(nearest_norm_points[:, 0], nearest_norm_points[:, 1], c=region_data, cmap='gist_heat', marker='o')
            plt.quiver(grid[0][flame_x_idx], grid[1][flame_y_idx], flame_norm[0], flame_norm[1],
                       angles='xy', scale_units='xy', scale=5, color='r', label='Normals')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.xlim(min(region_grid[:, 0].ravel()), max(region_grid[:, 0].ravel()))
            plt.ylim(min(region_grid[:, 1].ravel()), max(region_grid[:, 1].ravel()))

            formatted_time = '{:.16f}'.format(raw_data.current_time.to_value()).rstrip('0').rstrip('.')
            filename = os.path.join(output_dir_path, f"Flame-Thickness-Animation-Time-{formatted_time}.png")
            plt.savefig(filename, format='png')
            plt.show()
            plt.close()

            """
            plt.figure(figsize=(8, 6))
            plt.scatter(region["x"].to_value(), region["y"].to_value(),
                        c=region['Temp'].to_value().reshape(len(np.unique(region["x"].to_value())),
                                                            len(np.unique(region["y"].to_value()))), cmap='gist_heat',
                        marker='o')
            plt.plot(np.array(contour_arr[:, 0], dtype=float), np.array(contour_arr[:, 1], dtype=float), color='r')
            plt.quiver(grid[0][flame_x_idx][0], grid[1][flame_y_idx][0], flame_norm[0], flame_norm[1],
                       angles='xy', scale_units='xy', scale=5, color='r', label='Normals')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.xlim(min(region_grid[:, 0].ravel()), max(region_grid[:, 0].ravel()))
            plt.ylim(min(region_grid[:, 1].ravel()), max(region_grid[:, 1].ravel()))
            plt.show()
            """

        return flame_thickness_val

    """

    :return: flame length for 2D simulation data
    """
    # Step 1: Load the current plt file
    raw_data.force_periodicity()
    all_data = raw_data.all_data()
    verts = all_data.extract_isocontours("Temp", 2000)
    rough_index = np.lexsort((verts[:, 0], verts[:, 1]))
    rough_sort = verts[rough_index]

    # Step 2: Remove outliers, artifacts of the periodicity, and order the array
    # Let us determine the buffer region caused by the periodic boundaries required
    buffer = 0.0125 * raw_data.domain_right_edge.to_value()[1]
    y_lower = raw_data.domain_left_edge.to_value()[1] + buffer
    y_upper = raw_data.domain_right_edge.to_value()[1] - buffer
    # Handling 2D contours from extracted vertices
    # Assuming verts contains 3D coordinates (x, y, z), we'll extract the 2D data
    # verts is a list of arrays, each array contains (x, y, z) vertices
    contour_pts = np.empty((len(rough_sort), 2), dtype=object)
    for i, vert in enumerate(rough_sort):
        # Extract only x and y for 2D visualization for valid points
        if (not np.isclose(vert[0], 0, atol=1e-02)
                and not np.isclose(vert[0], raw_data.domain_right_edge[0].to_value(), atol=1e-04)
                and (y_lower <= vert[1] <= y_upper)):
            if i == 0:
                contour_pts[i, 0] = vert[0]
                contour_pts[i, 1] = vert[1]
            if i > 0:
                if (vert[0] - rough_sort[i - 1, 0]) != 0 and (vert[1] - rough_sort[i - 1, 1]) != 0:
                    contour_pts[i, 0] = vert[0]
                    contour_pts[i, 1] = vert[1]

    contour_pts = np.array([x for x in contour_pts if x[0] is not None])
    #
    contour_arr = sort_by_nearest_neighbors(contour_pts).astype(float)
    # Calculate the length of a line between adjacent points for the determine contour
    contour_segments = []
    current_contour = [contour_arr[0]]  # Start with the first point

    contour_lines = []
    current_line = []

    for i in range(1, len(contour_arr)):
        temp_var = np.sqrt((contour_arr[i, 0] - contour_arr[i - 1, 0]) ** 2 +
                           (contour_arr[i, 1] - contour_arr[i - 1, 1]) ** 2)

        if temp_var > 0.1 * raw_data.domain_right_edge[1].to_value():
            # If the distance exceeds the threshold, start a new contour line
            contour_segments.append(np.array(current_contour))
            current_contour = [contour_arr[i]]  # Start new contour with the current point

            contour_lines.append(np.array(current_line))
            current_line = []
        else:
            current_contour.append(contour_arr[i])
            current_line.append(temp_var)

    # Append the last contour line if not empty
    if current_contour:
        contour_segments.append(np.array(current_contour))
        contour_lines.append(np.array(current_line))

    # Calculate the total line length, by summing the lines between points
    surface_length = sum(np.sum(distances) for distances in contour_lines)

    # Determine the normal vectors to the flame surface
    flame_thickness_val = flame_thickness(raw_data, contour_arr, center_loc, grid)

    if contour_check and thickness_check:
        return surface_length / 100, flame_thickness_val
    elif contour_check:
        return surface_length / 100
    elif thickness_check:
        return flame_thickness_val


def state_animation(method, **kwargs):
    def plot_frame(reference_loc=None):
        def plot_axis(ax, x_data, y_data, label, linestyle, color, ylabel, ylim, log_scale=False):
            """Helper function to plot data on a single axis."""
            ax.plot(x_data, y_data, label=label, linestyle=linestyle, color=color)
            ax.set_ylabel(ylabel)
            ax.set_ylim(*ylim)
            if log_scale:
                ax.set_yscale('log')
            # Automatically choose the best location for the legend
            ax.legend(loc='best')

        # Initialize plot
        fig, ax1 = plt.subplots()
        x_data_full = plt_data["boxlib", "x"][sort_arr].to_value()

        if isinstance(plt_type, (np.ndarray, list, tuple)):
            axes = [ax1]
            for i, obj in enumerate(pelec_key):
                # Prepare x and y data
                tmp_x_data = x_data_full
                tmp_y_data = hrr_arr if plt_type[i] in ['HRR-Cantera', 'HRR-PeleC'] else plt_data["boxlib", obj][
                    sort_arr].to_value()

                # Apply reference location filter if provided
                if reference_loc is not None:
                    indices = np.where(
                        (tmp_x_data >= x_data_full[reference_loc] - reference_loc / 2) &
                        (tmp_x_data <= x_data_full[reference_loc] + reference_loc / 2)
                    )[0]
                    tmp_x_data = tmp_x_data[indices]
                    tmp_y_data = tmp_y_data[indices]

                # Plot primary axis or create secondary axes
                if i == 0:
                    plot_axis(
                        ax=axes[0],
                        x_data=tmp_x_data,
                        y_data=tmp_y_data,
                        label=obj,
                        linestyle='-',
                        color='k',
                        ylabel=obj,
                        ylim=(mim_bnd_val[i], max_bnd_val[i])
                    )
                    ax1.set_xlabel('Position [cm]')
                    ax1.grid(True, axis='x')
                else:
                    ax = ax1.twinx()
                    ax.spines["right"].set_position(("outward", 60 * i))  # Offset for clarity
                    plot_axis(
                        ax=ax,
                        x_data=tmp_x_data,
                        y_data=tmp_y_data,
                        label=obj,
                        linestyle='--',
                        color=f"C{i}",  # Automatically cycle colors
                        ylabel=obj,
                        ylim=(mim_bnd_val[i], max_bnd_val[i]),
                        log_scale=(obj == "pressure")
                    )
                    axes.append(ax)

            # Title
            title = " and ".join(plt_type) + f" Variation at y = {domain_size[1][1][1]} cm and t = {time} s"
        else:
            tmp_y_data = hrr_arr if plt_type in ['HRR-Cantera', 'HRR-PeleC'] else plt_data["boxlib", pelec_key][
                sort_arr].to_value()

            # Apply reference location filter if provided
            if reference_loc is not None:
                indices = np.where(
                    (x_data_full >= x_data_full[reference_loc] - reference_loc / 2) &
                    (x_data_full <= x_data_full[reference_loc] + reference_loc / 2)
                )[0]
                tmp_x_data = x_data_full[indices]
                tmp_y_data = tmp_y_data[indices]
            else:
                tmp_x_data = x_data_full

            plot_axis(
                ax=ax1,
                x_data=tmp_x_data,
                y_data=tmp_y_data,
                label=plt_type,
                linestyle='-',
                color='k',
                ylabel=plt_type,
                ylim=(mim_bnd_val, max_bnd_val),
                log_scale=(plt_type == "pressure")
            )

            # Title
            title = f"{plt_type} Variation at y = {domain_size[1][1][1]} cm and t = {time} s"

            # Wrap and set title
        wrapped_title = "\n".join(textwrap.wrap(title, width=55))
        plt.suptitle(wrapped_title, ha="center")

        # Set x-axis limits and legend
        ax1.set_xlim(0, domain_size[1][1][0])

        # Adjust layout to avoid overlapping with the legend
        plt.tight_layout()

        # Format time for filename
        formatted_time = f"{time:.16f}".rstrip('0').rstrip('.')

        # Create filename and save plot
        if isinstance(plt_type, (np.ndarray, list, tuple)):
            filename = os.path.join(output_dir_path, f"{'-'.join(plt_type)}-Animation-Time-{formatted_time}.png")
        else:
            filename = os.path.join(output_dir_path, f"{plt_type}-Animation-Time-{formatted_time}.png")

        plt.savefig(filename, format='png')
        plt.close()  # Avoid displaying inline in notebooks

        return

    def create_animation():

        # Step 1:
        image_files = [os.path.join(folder_path, f) for f in sorted(os.listdir(folder_path)) if
                       f.endswith(f'.png')]

        # Step 2: Load the first image to determine the size
        first_image = mpimg.imread(image_files[0])

        # Step 3: Create a figure with no axes
        fig = plt.figure(figsize=(first_image.shape[1] / 100, first_image.shape[0] / 100), dpi=100)
        plt.axis('off')

        # Step 4: Placeholder for the image
        img_display = plt.imshow(first_image)

        # Step 5: Update function for animation
        def update(frame):
            img_display.set_array(mpimg.imread(image_files[frame]))
            return img_display,

        # Step 6: Create animation
        ani = animation.FuncAnimation(fig, update, frames=len(image_files), blit=True)

        # Step 7: Configure writer and save animation
        writer = animation.FFMpegWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)
        # writer = animation.writers['ffmpeg']
        # writer = writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

        ani.save(animation_filename, writer=writer)

        plt.close(fig)  # Close the figure

        return

    """

    """
    # Step 0: Get values from **kwargs
    plt_type = kwargs.get('plt_type', [])
    mim_bnd_val = kwargs.get('mim_bnd_val', 0)
    max_bnd_val = kwargs.get('max_bnd_val', 0)

    reference_loc = kwargs.get('reference_loc', None)  # Get reference_loc from kwargs
    time = kwargs.get('time', 0)
    plt_data = kwargs.get('plt_data', [])
    sort_arr = kwargs.get('sort_arr', [])
    pelec_key = kwargs.get('pelec_key', '')
    domain_size = kwargs.get('domain_size', [0, 0])
    hrr_arr = kwargs.get('hrr_arr', [])

    folder_path = kwargs.get('folder_path', "")
    output_dir_path = kwargs.get('output_dir_path', "")
    animation_filename = kwargs.get('animation_filename', "")

    # Step 1: Set the plotting method (domain or local) or animation creation
    if method == 'Plot':
        if reference_loc is None:
            plot_frame()
        else:
            plot_frame(reference_loc=reference_loc)

    elif method == 'Animate':
        create_animation()
    else:
        print('Error: Did not define viable method argument (Plot or Animate)')

    return


def animation_bounds(args):
    iter_val, const_list, input_params = args

    temp_plt_files = iter_val
    domain_info = const_list[0]
    parameter_dict = const_list[1]
    # Step 1: Create an array of the keys where the value is True
    keys_with_true_values = [
        key for key, value in parameter_dict['Domain State Animations'].items()
        if isinstance(value, tuple) and value[0] is True
    ]

    # Step 2: Initialize the temp_max_val_arr with the same length as the number of True values in the dictionary
    temp_bounds_arr = np.empty((len(keys_with_true_values), 3), dtype=object)

    # Step 3: Load data and collect the relevent parameters
    raw_data = yt.load(temp_plt_files)
    slice = raw_data.ray(np.array([domain_info[1][0][0], domain_info[1][0][1], 0.0]),
                         np.array([domain_info[1][1][0], domain_info[1][1][1], 0.0]))

    for i, key in enumerate(keys_with_true_values):
        if key == 'Temperature':
            temp_arr = slice["boxlib", "Temp"].to_value()

        if key == 'Pressure':
            temp_arr = slice["boxlib", "pressure"].to_value()

        if key == 'Velocity':
            temp_arr = slice["boxlib", "x_velocity"].to_value()

        if key == 'Species':
            print('Max Value Determination for Species is W.I.P.')

        if key == 'Heat Release Rate Cantera':
            temp_arr = heat_release_rate_function(slice, np.argsort(slice["boxlib", "x"]), input_params)[0]

        if key == 'Heat Release Rate PeleC':
            try:
                temp_arr = slice["boxlib", "heatRelease"].to_value()
            except:
                temp_arr = heat_release_rate_function(slice, np.argsort(slice["boxlib", "x"]), input_params)[0]

        if key == 'Flame Thickness':
            temp_arr = flame_geometry_function(raw_data, domain_info[1][0][1], domain_info[-1], thickness_check=True)

        # Step 3: Write bounds to value
        temp_bounds_arr[i, 0] = key
        temp_bounds_arr[i, 1] = np.min(temp_arr)
        temp_bounds_arr[i, 2] = np.max(temp_arr)

    return temp_bounds_arr


def createVariablePltFrame(raw_data, sort_arr, time, min_bounds, max_bounds, tracking_obj, domain_info, output_path,
                           hrr_arr=None):
    def pltAnimationFrame():
        """
            Plots the data and saves the plot as an image file.

            Parameters:
            - x: array-like, the data for the x-axis.
            - y: array-like, the data for the y-axis.
            - filename: str, the name of the file to save the plot.
            - xlabel: str, label for the x-axis (default is 'X-axis').
            - ylabel: str, label for the y-axis (default is 'Y-axis').
            - title: str, title of the plot (default is 'Plot').
            - format: str, the format to save the plot in (default is 'png').
            """
        line_styles = ['-', '--', '-.', ':']
        style_cycle = itertools.cycle(line_styles)

        fig, ax1 = plt.subplots()
        if isinstance(tracking_obj, (np.ndarray, list)):
            ax1.plot(raw_data["boxlib", str('x')][sort_arr].to_value(),
                     raw_data["boxlib", str(tracking_obj[0])][sort_arr].to_value(),
                     label=tracking_obj[0], linestyle='-', color='k')
            ax1.set_xlabel('Position [cm]')
            ax1.set_ylabel(str(tracking_obj[0]))
            ax1.grid(True, axis='x')  # Only x-axis grid lines
            ax1.set_ylim(y_limit_min[0], y_limit_max[0])

            # Create a second Y-axis
            ax2 = ax1.twinx()
            ax2.plot(raw_data["boxlib", str('x')][sort_arr].to_value(),
                     raw_data["boxlib", str(tracking_obj[1])][sort_arr].to_value(),
                     label=tracking_obj[1], linestyle='--', color='r')
            if tracking_obj[1] == "pressure":
                plt.yscale('log')
            ax2.set_ylabel(str(tracking_obj[1]))
            ax2.set_ylim(y_limit_min[1], y_limit_max[1])

            wrapped_title = "\n".join(textwrap.wrap(
                f"{tracking_obj[0]} and {tracking_obj[1]} Variation at y = {domain_size[1]} and t = {time}", width=55))
        else:
            if tracking_obj == 'HRR-Cantera' or tracking_obj == 'HRR-PeleC':
                ax1.plot(raw_data["boxlib", str('x')][sort_arr].to_value(), hrr_arr,
                         label=tracking_obj, linestyle='-', color='k')
            else:
                ax1.plot(raw_data["boxlib", str('x')][sort_arr].to_value(),
                         raw_data["boxlib", str(tracking_obj)][sort_arr].to_value(),
                         label=tracking_obj, linestyle='-', color='k')
                if tracking_obj == "pressure":
                    plt.yscale('log')

            ax1.set_ylabel(str(tracking_obj))
            ax1.set_ylim(y_limit_min, y_limit_max)

            wrapped_title = "\n".join(
                textwrap.wrap(f"{tracking_obj} Variation at y = {domain_size[1]} and t = {time}", width=55))

        plt.suptitle(wrapped_title, ha="center")
        plt.xlim(0, domain_size[0])
        plt.legend()

        # Convert the float to a string without scientific notation
        formatted_time = '{:.16f}'.format(time).rstrip('0').rstrip('.')
        # Save the plot to a file
        if isinstance(tracking_obj, (np.ndarray, list)):
            filename = os.path.join(output_dir_path,
                                    f"{tracking_obj[0]}-{tracking_obj[1]}-Animation-Time-{formatted_time}.png")
        else:
            filename = os.path.join(output_dir_path, f"{tracking_obj}-Animation-Time-{formatted_time}.png")
        plt.savefig(filename, format='png')
        plt.close()  # Close the figure to avoid displaying it inline if using in a Jupyter notebook
        return

    """

    """
    # Step 1:
    if isinstance(tracking_obj, (np.ndarray, list)):
        y_limit_min = np.zeros(len(tracking_obj))
        y_limit_max = np.zeros(len(tracking_obj))
        for i in range(len(tracking_obj)):
            y_limit_min[i] = min_bounds[i] - (0.1 * min_bounds[i])
            y_limit_max[i] = max_bounds[i] + (0.1 * max_bounds[i])
    else:
        y_limit_min = min_bounds - (0.1 * min_bounds)
        y_limit_max = max_bounds + (0.1 * max_bounds)

    # Step 2:
    domain_size = domain_info[1][1]
    # Step 3: Check/Create directory to store animation frmaes
    if isinstance(tracking_obj, (np.ndarray, list)):
        output_dir_path = ensure_long_path_prefix(os.path.join(output_path, f"Animation-Frames",
                                                               f"{tracking_obj[0]}-{tracking_obj[1]}-Plt-Files"))
    else:
        output_dir_path = ensure_long_path_prefix(os.path.join(output_path, f"Animation-Frames",
                                                               f"{tracking_obj}-Plt-Files"))

    # Step 3:
    pltAnimationFrame()

    return


def createVariableAnimation(folder_path, output_filename, fps=15):
    def collect_plot_files(folder_path, file_extension='png'):
        """
        Collects all plot files from a given folder.

        Parameters:
        - folder_path: str, the path to the folder containing plot files.
        - file_extension: str, the file extension of the plot files (default is 'png').

        Returns:
        - image_files: list of str, paths to the image files sorted by filename.
        """
        # List all files in the directory with the given extension
        image_files = [os.path.join(folder_path, f) for f in sorted(os.listdir(folder_path)) if
                       f.endswith(f'.{file_extension}')]
        return image_files

    def create_animation(image_files, output_filename, fps=2):
        """
        Creates an MP4 animation from a series of image files.

        Parameters:
        - image_files: list of str, paths to the image files.
        - output_filename: str, the name of the output video file (e.g., 'animation.mp4').
        - fps: int, frames per second for the animation (default is 2).
        """
        # Create a figure and axis
        fig, ax = plt.subplots()

        # Load the first image to get the size
        first_image = mpimg.imread(image_files[0])
        ax.set_xlim(0, first_image.shape[1])
        ax.set_ylim(0, first_image.shape[0])

        # Placeholder for the image
        img_display = ax.imshow(first_image)

        # Update function for animation
        def update(frame):
            img_display.set_array(np.flipud(mpimg.imread(image_files[frame])))
            return img_display,

        # Create animation
        ani = animation.FuncAnimation(fig, update, frames=len(image_files), blit=True)

        # Choose the writer and save the animation
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=fps, metadata=dict(artist='Me'), bitrate=1800)

        # Save the animation as MP4
        ani.save(output_filename, writer=writer)

        plt.close()  # Close the plot

    """
        Creates an MP4 animation from plot files in a given folder.

        Parameters:
        - folder_path: str, the path to the folder containing plot files.
        - output_filename: str, the name of the output video file (e.g., 'animation.mp4').
        - fps: int, frames per second for the animation (default is 2).
        - file_extension: str, the file extension of the plot files (default is 'png').
    """
    # Collect all plot files from the folder
    image_files = collect_plot_files(folder_path, 'png')

    # Create animation using the collected plot files
    create_animation(image_files, output_filename, fps)
    return


def single_pltfile_processing(args):
    def load_data():
        raw_data = yt.load(pltFile_dir)
        slice = raw_data.ray(np.array([domain_info[1][0][0], domain_info[1][0][1], 0.0]),
                             np.array([domain_info[1][1][0], domain_info[1][1][1], 0.0]))
        ray_sort = np.argsort(slice["boxlib", "x"])
        time = raw_data.current_time.to_value()

        plt_check = False
        if plt_check:
            plt.figure(figsize=(8, 6))
            plt.plot(slice["boxlib", "x"][ray_sort], slice["boxlib", "Temp"][ray_sort], linestyle='-', color='k')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.show()

        return time, raw_data, slice, ray_sort

    """

    """
    # Step 1: Unpack the argumnents provided
    iter_val, const_list, input_params = args

    pltFile_dir = iter_val
    processing_flags = const_list[0]
    animation_bnd_arr = const_list[1]
    domain_info = const_list[2]
    output_path = const_list[3]

    # Step 2: Load the pltFile for individual processing
    time, raw_data, plt_data, sort_arr = load_data()

    # Step 3:
    result_dict = {}
    result_dict['Time'] = {}
    result_dict['Time']['Value'] = time

    # Step 3.1: Flame Processing
    if 'Flame Processing' in processing_flags:
        result_dict['Flame'] = {}

        # Position
        if processing_flags['Flame Processing'].get('Position', False):
            [result_dict['Flame']['Index'], result_dict['Flame']['Position']] = wave_tracking_function(plt_data,
                                                                                                       sort_arr, "Temp",
                                                                                                       "Flame")

        # Gas Velocity
        if processing_flags['Flame Processing'].get('Relative Velocity', False):
            result_dict['Flame']['Gas Velocity'] = plt_data["boxlib", "x_velocity"][sort_arr][
                                                       result_dict['Flame']['Index'] + 10].to_value() / 100

        # Thermodynamic State
        if processing_flags['Flame Processing'].get('Thermodynamic State', False):
            result_dict['Flame']['Thermodynamic State'] = thermodynamic_state_function(plt_data, sort_arr,
                                                                                       result_dict['Flame']['Index'],
                                                                                       "Flame", input_params)

        # Flame Heat Release Rate
        if processing_flags['Flame Processing'].get('Heat Release Rate Cantera', False):
            [result_dict['Flame']['Heat Release Rate Cantera'],
             result_dict['Flame']['Max Heat Release Rate Cantera']] = heat_release_rate_function(plt_data, sort_arr,
                                                                                                 input_params)

        if processing_flags['Flame Processing'].get('Heat Release Rate PeleC', False):
            try:
                result_dict['Flame']['Heat Release Rate PeleC'] = plt_data["boxlib", "heatRelease"][sort_arr].to_value()
                result_dict['Flame']['Max Heat Release Rate PeleC'] = max(
                    result_dict['Flame']['Heat Release Rate PeleC'])
            except:
                [result_dict['Flame']['Heat Release Rate PeleC'],
                 result_dict['Flame']['Max Heat Release Rate PeleC']] = heat_release_rate_function(plt_data, sort_arr,
                                                                                                   input_params)

        # Surface Flame Thickness
        if processing_flags['Flame Processing'].get('Flame Thickness', False) or processing_flags[
            'Flame Processing'].get('Surface Length', False):
            temp_plt_dir = ensure_long_path_prefix(
                os.path.join(output_path, f"Animation-Frames", f"Flame-Thickness-Plt-Files"))

            if os.path.exists(temp_plt_dir) is False:
                os.makedirs(temp_plt_dir, exist_ok=True)
            try:
                if processing_flags['Flame Processing'].get('Flame Thickness', False) and processing_flags[
                    'Flame Processing'].get('Surface Length', False):
                    [result_dict['Flame']['Surface Length'],
                     result_dict['Flame']['Flame Thickness']] = flame_geometry_function(raw_data,
                                                                                        domain_info[1][0][1],
                                                                                        domain_info[-1],
                                                                                        temp_plt_dir,
                                                                                        thickness_check=True,
                                                                                        contour_check=True)

                # Flame Thickness
                if processing_flags['Flame Processing'].get('Flame Thickness', False) and not processing_flags[
                    'Flame Processing'].get('Surface Length', False):
                    result_dict['Flame']['Flame Thickness'] = flame_geometry_function(raw_data,
                                                                                      domain_info[1][0][1],
                                                                                      domain_info[-1],
                                                                                      temp_plt_dir,
                                                                                      thickness_check=True)

                if processing_flags['Flame Processing'].get('Surface Length', False) and not processing_flags[
                    'Flame Processing'].get('Flame Thickness', False):
                    result_dict['Flame']['Surface Length'] = flame_geometry_function(raw_data,
                                                                                     domain_info[1][0][1],
                                                                                     domain_info[-1],
                                                                                     None,
                                                                                     contour_check=True)
            except:
                result_dict['Flame']['Surface Length'] = 0
                result_dict['Flame']['Flame Thickness'] = 0

    # Step 3.2: Maximum Pressure Processing
    if 'Maximum Pressure Processing' in processing_flags:
        result_dict['Maximum Pressure'] = {}

        # Position
        if processing_flags['Maximum Pressure Processing'].get('Position', False):
            [result_dict['Max Pressure']['Index'], result_dict['Max Pressure']['Position']] = wave_tracking_function(
                plt_data, sort_arr, "presssure", "Maximum Pressure")

        # Thermodynamic State
        if processing_flags['Maximum Pressure Processing'].get('Thermodynamic State', False):
            result_dict['Max Pressure']['Thermodynamic State'] = thermodynamic_state_function(plt_data, sort_arr,
                                                                                              result_dict[
                                                                                                  'Max Pressure'][
                                                                                                  'Index'],
                                                                                              "Maximum Pressure",
                                                                                              input_params)

    # Step 3.2: Leading Shock Wave Processing
    if 'Leading Shock Processing' in processing_flags:
        result_dict['Lead Shock'] = {}

        # Position
        if processing_flags['Leading Shock Processing'].get('Position', False):
            [result_dict['Lead Shock']['Index'], result_dict['Lead Shock']['Position']] = wave_tracking_function(
                plt_data, sort_arr, "pressure", "Leading Shock")

    # Step 3.3: Pre-Shock Processing
    if 'Pre-Shock Processing' in processing_flags:
        result_dict['Pre-Shock'] = {}

        if processing_flags['Pre-Shock Processing'].get('Thermodynamic State', False):
            # If the leading shock wave is not flagged, determine the location here
            if not processing_flags['Leading Shock Processing'].get('Position', False):
                result_dict['Lead Shock'] = {}
                [result_dict['Lead Shock']['Index'], _] = wave_tracking_function(plt_data, sort_arr, "pressure",
                                                                                 "Leading Shock")

            # Determine the thermodynamic state ahead of the leading shock wave
            result_dict['Pre-Shock']['Thermodynamic State'] = thermodynamic_state_function(plt_data, sort_arr,
                                                                                           result_dict['Lead Shock'][
                                                                                               'Index'], "Pre-Shock",
                                                                                           input_params)

    # Step 3.4: Post-Shock Processing
    if 'Post-Shock Processing' in processing_flags:
        result_dict['Post-Shock'] = {}

        if processing_flags['Post-Shock Processing'].get('Thermodynamic State', False):
            # If the leading shock wave is not flagged, determine the location here
            if not processing_flags['Leading Shock Processing'].get('Position', False):
                result_dict['Lead Shock'] = {}
                [result_dict['Lead Shock']['Index'], _] = wave_tracking_function(plt_data, sort_arr, "pressure",
                                                                                 "Leading Shock")

            # Determine the thermodynamic state behind of the leading shock wave
            result_dict['Post-Shock']['Thermodynamic State'] = thermodynamic_state_function(plt_data, sort_arr,
                                                                                            result_dict['Lead Shock'][
                                                                                                'Index'], "Post-Shock",
                                                                                            input_params)

    # Step 3.5: State Variation Animation
    if 'Domain State Animations' in processing_flags:
        plot_check = processing_flags.get('Domain State Animations', {})

        for key, value in plot_check.items():
            if key == 'Combined':
                min_bnd_val = []
                max_bnd_val = []
                for i in range(len(value[0])):
                    # Collect the plot bounds based on the current desire
                    bnd_arr_index = np.where(animation_bnd_arr[:, 0] == value[0][i])[0][
                        0]  # Get the first matching index
                    min_bnd_val.append(animation_bnd_arr[bnd_arr_index, 1])
                    max_bnd_val.append(animation_bnd_arr[bnd_arr_index, 2])

                hrr_arr = None
                if key == 'Heat Release Rate Cantera' or key == 'Heat Release Rate PeleC':
                    hrr_arr = result_dict['Flame'][value[0]]

                # Create directory for plt files
                combined_key = "-".join(value[0])
                temp_plt_dir = ensure_long_path_prefix(
                    os.path.join(output_path, f"Animation-Frames", f"{combined_key}-Plt-Files"))

                if os.path.exists(temp_plt_dir) is False:
                    os.makedirs(temp_plt_dir, exist_ok=True)

                state_animation(
                    method='Plot',
                    plt_type=value[0],
                    mim_bnd_val=min_bnd_val,
                    max_bnd_val=max_bnd_val,
                    reference_loc=None,
                    time=time,
                    plt_data=plt_data,
                    sort_arr=sort_arr,
                    pelec_key=value[1],
                    domain_size=domain_info,
                    hrr_arr=hrr_arr,
                    output_dir_path=temp_plt_dir,
                )

            else:
                if isinstance(value, (np.ndarray, list, tuple)):
                    bool_check = value[0]
                    pelec_key = value[1]
                else:
                    bool_check = value
                    pelec_key = None

                if bool_check:
                    # Collect the plot bounds based on the current desire
                    bnd_arr_index = np.where(animation_bnd_arr[:, 0] == key)[0][0]  # Get the first matching index
                    min_bnd_val = animation_bnd_arr[bnd_arr_index, 1]
                    max_bnd_val = animation_bnd_arr[bnd_arr_index, 2]

                    # Create directory for plt files
                    temp_plt_dir = ensure_long_path_prefix(
                        os.path.join(output_path, f"Animation-Frames", f"{key}-Plt-Files"))

                    if os.path.exists(temp_plt_dir) is False:
                        os.makedirs(temp_plt_dir, exist_ok=True)

                    hrr_arr = None
                    if key == 'Heat Release Rate Cantera' or key == 'Heat Release Rate PeleC':
                        hrr_arr = result_dict['Flame'][key]

                    state_animation(
                        method='Plot',
                        plt_type=key,
                        mim_bnd_val=min_bnd_val,
                        max_bnd_val=max_bnd_val,
                        reference_loc=None,
                        time=time,
                        plt_data=plt_data,
                        sort_arr=sort_arr,
                        pelec_key=pelec_key,
                        domain_size=domain_info,
                        hrr_arr=hrr_arr,
                        output_dir_path=temp_plt_dir,
                    )

        """
        # Step 3.5.3: Plot Temperature
        if processing_flags['Domain State Animations'].get('Temperature', False):
            # Select appropriate bounds
            index = np.where(animation_bnd_arr[:, 0] == 'Temperature')[0][0]  # Get the first matching index

            # Extract min and max values for the key
            min_val = animation_bnd_arr[index, 1]
            max_val = animation_bnd_arr[index, 2]

            # Create directory for plt files
            temp_plt_dir = ensure_long_path_prefix(os.path.join(output_path, f"Animation-Frames", f"Temp-Plt-Files"))

            if os.path.exists(temp_plt_dir) is False:
                os.makedirs(temp_plt_dir, exist_ok=True)

            createVariablePltFrame(plt_data, sort_arr, time, min_val, max_val, "Temp", domain_info, output_path)

        # Step 3.5.4: Plot Pressure
        if processing_flags['Domain State Animations'].get('Pressure', False):
            # Select appropriate bounds
            index = np.where(animation_bnd_arr[:, 0] == 'Pressure')[0][0]  # Get the first matching index

            # Extract min and max values for the key
            min_val = animation_bnd_arr[index, 1]
            max_val = animation_bnd_arr[index, 2]

            # Create directory for plt files
            temp_plt_dir = ensure_long_path_prefix(
                os.path.join(output_path, f"Animation-Frames", f"pressure-Plt-Files"))

            if os.path.exists(temp_plt_dir) is False:
                os.makedirs(temp_plt_dir, exist_ok=True)

            createVariablePltFrame(plt_data, sort_arr, time, min_val, max_val, "pressure", domain_info, output_path)

        # Step 3.5.5: Plot Velocity
        if processing_flags['Domain State Animations'].get('Velocity', False):
            # Select appropriate bounds
            index = np.where(animation_bnd_arr[:, 0] == 'Velocity')[0][0]  # Get the first matching index

            # Extract min and max values for the key
            min_val = animation_bnd_arr[index, 1]
            max_val = animation_bnd_arr[index, 2]

            # Create directory for plt files
            temp_plt_dir = ensure_long_path_prefix(os.path.join(output_path, f"Animation-Frames", f"x_velocity-Plt-Files"))

            if os.path.exists(temp_plt_dir) is False:
                os.makedirs(temp_plt_dir, exist_ok=True)

            createVariablePltFrame(plt_data, sort_arr, time, min_val, max_val, "x_velocity", domain_info, output_path)

        # Step 3.5.5: Plot Velocity
        if processing_flags['Domain State Animations'].get('Heat Release Rate Cantera', False):
            # Select appropriate bounds
            index = np.where(animation_bnd_arr[:, 0] == 'Heat Release Rate Cantera')[0][0]  # Get the first matching index

            # Extract min and max values for the key
            min_val = animation_bnd_arr[index, 1]
            max_val = animation_bnd_arr[index, 2]

            # Create directory for plt files
            temp_plt_dir = ensure_long_path_prefix(os.path.join(output_path, f"Animation-Frames", f"HRR-Cantera-Plt-Files"))

            if os.path.exists(temp_plt_dir) is False:
                os.makedirs(temp_plt_dir, exist_ok=True)

            createVariablePltFrame(plt_data, sort_arr, time, min_val, max_val, "HRR-Cantera",
                                   domain_info, output_path, hrr_arr = result_dict['Flame']['Heat Release Rate Cantera'])

        if processing_flags['Domain State Animations'].get('Heat Release Rate PeleC', False):
            # Select appropriate bounds
            index = np.where(animation_bnd_arr[:, 0] == 'Heat Release Rate PeleC')[0][0]  # Get the first matching index

            # Extract min and max values for the key
            min_val = animation_bnd_arr[index, 1]
            max_val = animation_bnd_arr[index, 2]

            # Create directory for plt files
            temp_plt_dir = ensure_long_path_prefix(os.path.join(output_path,f"Animation-Frames", f"HRR-PeleC-Plt-Files"))

            if os.path.exists(temp_plt_dir) is False:
                os.makedirs(temp_plt_dir, exist_ok=True)

            createVariablePltFrame(plt_data, sort_arr, time, min_val, max_val, "HRR-PeleC",
                                   domain_info, output_path, hrr_arr = result_dict['Flame']['Heat Release Rate PeleC'])

        # Step 3.5.6: Plot Species
        if processing_flags['Domain State Animations'].get('Species', False):
            # Select appropriate bounds
            index = np.where(animation_bnd_arr[:, 0] == 'Species')[0][0]  # Get the first matching index

            # Extract min and max values for the key
            min_val = animation_bnd_arr[index, 1]
            max_val = animation_bnd_arr[index, 2]

            for i in range(len(input_params.species)):
                # Create directory for plt files
                temp_plt_dir = ensure_long_path_prefix(os.path.join(output_path, f"Animation-Frames", f"Y({str(input_params.species[i])})-Plt-Files"))
                if os.path.exists(temp_plt_dir) is False:
                    os.makedirs(temp_plt_dir, exist_ok=True)

                createVariablePltFrame(plt_data, sort_arr, time, min_val[i + 4], max_val[i + 4], f"Y({str(input_params.species[i])})", domain_info, output_path)

        # Step 3.5.3: Plot Combined
        if processing_flags['Domain State Animations'].get('Combined', False):
            # Select appropriate bounds
            index_1 = np.where(animation_bnd_arr[:, 0] == f"{processing_flags['Domain State Animations']['Combined'][0]}")[0][0]  # Get the first matching index
            index_2 = np.where(animation_bnd_arr[:, 0] == f"{processing_flags['Domain State Animations']['Combined'][2]}")[0][0]  # Get the first matching index

            # Extract min and max values for the key
            min_val_1 = animation_bnd_arr[index_1, 1]
            max_val_1 = animation_bnd_arr[index_1, 2]

            min_val_2 = animation_bnd_arr[index_2, 1]
            max_val_2 = animation_bnd_arr[index_2, 2]

            # Create directory for plt files
            temp_plt_dir = ensure_long_path_prefix(
                os.path.join(output_path, f"Animation-Frames", f"{processing_flags['Domain State Animations']['Combined'][0]}-"
                                                               f"{processing_flags['Domain State Animations']['Combined'][1]}-Plt-Files"))
            if os.path.exists(temp_plt_dir) is False:
                os.makedirs(temp_plt_dir, exist_ok=True)

            createVariablePltFrame(plt_data, sort_arr, time,
                                   [min_val_1, min_val_2], [max_val_1, max_val_2],
                                   [processing_flags['Domain State Animations']['Combined'][0],
                                    processing_flags['Domain State Animations']['Combined'][1]], domain_info, output_path)
        """

    return result_dict


def pelec_processing_function(plt_dir, domain_info, animation_bnd_arr, check_flags, output_dir):
    def file_output(file_path, smoothing_check):
        # Step 1: Dynamically create the text file header depending on the assigned flags
        header_data = ["Time [s]"]
        # Position
        if check_flags['Flame Processing'].get('Position', False):
            header_data.extend(["Flame Position [m]"])
        if check_flags['Leading Shock Processing'].get('Position', False):
            header_data.extend(["Leading Shock Position [m]"])
        if check_flags['Maximum Pressure Processing'].get('Position', False):
            header_data.extend(["Max Pressure Position [m]"])
        # Velocity
        if check_flags['Flame Processing'].get('Velocity', False):
            header_data.extend(["Flame Velocity [m/s]"])
        if check_flags['Leading Shock Processing'].get('Velocity', False):
            header_data.extend(["Leading Shock Velocity [m/s]"])
        # Relative Velocity
        if check_flags['Flame Processing'].get('Relative Velocity', False):
            header_data.extend(["Relative Flame Velocity [m/s]"])
        # Thermodynamic State
        if check_flags['Flame Processing'].get('Thermodynamic State', False):
            header_data.extend(["Flame Temperature [K]", "Flame Pressure [Pa]", "Flame Density [kg/m^3]",
                                "Flame Soundspeed [m/s]"])
        if check_flags['Maximum Pressure Processing'].get('Thermodynamic State', False):
            header_data.extend(["Max Pressure Temperature [K]", "Max Pressure Pressure [Pa]",
                                "Max Pressure Density [kg/m^3]", "Max Pressure Soundspeed [m/s]"])
        if check_flags['Pre-Shock Processing'].get('Thermodynamic State', False):
            header_data.extend(["Pre-Shock Temperature [K]", "Pre-Shock Pressure [Pa]",
                                "Pre-Shock Density [kg/m^3]", "Pre-Shock Soundspeed [m/s]"])
        if check_flags['Post-Shock Processing'].get('Thermodynamic State', False):
            header_data.extend(["Post-Shock Temperature [K]", "Post-Shock Pressure [Pa]",
                                "Post-Shock Density [kg/m^3]", "Post-Shock Soundspeed [m/s]"])
        # Heat Release Rate
        if check_flags['Flame Processing'].get('Heat Release Rate Cantera', False):
            header_data.extend(["Max Heat Release Rate Cantera [W/m3]"])
        if check_flags['Flame Processing'].get('Heat Release Rate PeleC', False):
            header_data.extend(["Max Heat Release Rate PeleC [egs/cm3]"])
        # Flame Thickness
        if check_flags['Flame Processing'].get('Flame Thickness', False):
            header_data.extend(["Flame Thickness [m]"])
        # Flame Surface Length
        if check_flags['Flame Processing'].get('Surface Length', False):
            header_data.extend(["Flame Surface Length [m]"])

        with open(file_path, "w") as outfile:
            outfile.write("#")
            for i in range(len(header_data)):
                outfile.write("{0:<55.0f} ".format(int(i + 1)))
            outfile.write("\n#")
            for i in range(len(header_data)):
                outfile.write("{0:<55s} ".format(header_data[i]))
            outfile.write("\n")

            if not smoothing_check:
                for i in range(len(collective_results['Time']['Value'])):
                    # Time
                    outfile.write(" {0:<55e}".format(collective_results['Time']['Value'][i]))

                    # Position
                    if check_flags['Flame Processing'].get('Position', False):
                        outfile.write(" {0:<55e}".format(collective_results['Flame']['Position'][i]))

                    if check_flags['Maximum Pressure Processing'].get('Position', False):
                        outfile.write(" {0:<55e}".format(collective_results['Maximum Pressure']['Position'][i]))

                    if check_flags['Leading Shock Processing'].get('Position', False):
                        outfile.write(" {0:<55e}".format(collective_results['Lead Shock']['Position'][i]))

                    # Velocity
                    if check_flags['Flame Processing'].get('Velocity', False):
                        outfile.write(" {0:<55e}".format(collective_results['Flame']['Velocity'][i]))

                    if check_flags['Leading Shock Processing'].get('Velocity', False):
                        outfile.write(" {0:<55e}".format(collective_results['Lead Shock']['Velocity'][i]))

                    # Relative Velocity
                    if check_flags['Flame Processing'].get('Relative Velocity', False):
                        outfile.write(" {0:<55e}".format(collective_results['Flame']['Relative Velocity'][i]))

                    # Thermodynamic State
                    if check_flags['Flame Processing'].get('Thermodynamic State', False):
                        for j in range(len(collective_results['Flame']['Thermodynamic State'][i])):
                            outfile.write(" {0:<55e}".format(collective_results['Flame']['Thermodynamic State'][i][j]))

                    if check_flags['Maximum Pressure Processing'].get('Thermodynamic State', False):
                        for j in range(len(collective_results['Maximum Pressure']['Thermodynamic State'][i])):
                            outfile.write(
                                " {0:<55e}".format(collective_results['Maximum Pressure']['Thermodynamic State'][i][j]))

                    if check_flags['Pre-Shock Processing'].get('Thermodynamic State', False):
                        for j in range(len(collective_results['Pre-Shock']['Thermodynamic State'][i])):
                            outfile.write(
                                " {0:<55e}".format(collective_results['Pre-Shock']['Thermodynamic State'][i][j]))

                    if check_flags['Post-Shock Processing'].get('Thermodynamic State', False):
                        for j in range(len(collective_results['Post-Shock']['Thermodynamic State'][i])):
                            outfile.write(
                                " {0:<55e}".format(collective_results['Post-Shock']['Thermodynamic State'][i][j]))

                    # Heat Release Rate
                    if check_flags['Flame Processing'].get('Heat Release Rate Cantera', False):
                        outfile.write(
                            " {0:<55e}".format(collective_results['Flame']['Max Heat Release Rate Cantera'][i]))

                    if check_flags['Flame Processing'].get('Heat Release Rate PeleC', False):
                        outfile.write(" {0:<55e}".format(collective_results['Flame']['Max Heat Release Rate PeleC'][i]))

                    # Flame Thickness
                    if check_flags['Flame Processing'].get('Flame Thickness', False):
                        outfile.write(" {0:<55e}".format(collective_results['Flame']['Flame Thickness'][i]))

                    # Surface Length
                    if check_flags['Flame Processing'].get('Surface Length', False):
                        outfile.write(" {0:<55e}".format(collective_results['Flame']['Surface Length'][i]))
                    outfile.write("\n")

            if smoothing_check:
                for i in range(len(collective_results['Time']['Smooth']['Value'])):
                    # Time
                    outfile.write(" {0:<55e}".format(collective_results['Time']['Smooth']['Value'][i]))

                    # Position
                    if check_flags['Flame Processing'].get('Position', False):
                        outfile.write(" {0:<55e}".format(collective_results['Flame']['Smooth']['Position'][i]))

                    if check_flags['Maximum Pressure Processing'].get('Position', False):
                        outfile.write(
                            " {0:<55e}".format(collective_results['Maximum Pressure']['Smooth']['Position'][i]))

                    if check_flags['Leading Shock Processing'].get('Position', False):
                        outfile.write(" {0:<55e}".format(collective_results['Lead Shock']['Smooth']['Position'][i]))

                    # Velocity
                    if check_flags['Flame Processing'].get('Velocity', False):
                        outfile.write(" {0:<55e}".format(collective_results['Flame']['Smooth']['Velocity'][i]))

                    if check_flags['Leading Shock Processing'].get('Velocity', False):
                        outfile.write(" {0:<55e}".format(collective_results['Lead Shock']['Smooth']['Velocity'][i]))

                    # Relative Velocity
                    if check_flags['Flame Processing'].get('Relative Velocity', False):
                        outfile.write(" {0:<55e}".format(collective_results['Flame']['Smooth']['Relative Velocity'][i]))

                    # Thermodynamic State
                    if check_flags['Flame Processing'].get('Thermodynamic State', False):
                        for j in range(len(collective_results['Flame']['Smooth']['Thermodynamic State'])):
                            outfile.write(
                                " {0:<55e}".format(collective_results['Flame']['Smooth']['Thermodynamic State'][j][i]))

                    if check_flags['Maximum Pressure Processing'].get('Thermodynamic State', False):
                        for j in range(len(collective_results['Maximum Pressure']['Smooth']['Thermodynamic State'])):
                            outfile.write(
                                " {0:<55e}".format(
                                    collective_results['Maximum Pressure']['Smooth']['Thermodynamic State'][j][i]))

                    if check_flags['Pre-Shock Processing'].get('Thermodynamic State', False):
                        for j in range(len(collective_results['Pre-Shock']['Smooth']['Thermodynamic State'])):
                            outfile.write(
                                " {0:<55e}".format(
                                    collective_results['Pre-Shock']['Smooth']['Thermodynamic State'][j][i]))

                    if check_flags['Post-Shock Processing'].get('Thermodynamic State', False):
                        for j in range(len(collective_results['Post-Shock']['Smooth']['Thermodynamic State'])):
                            outfile.write(
                                " {0:<55e}".format(
                                    collective_results['Post-Shock']['Smooth']['Thermodynamic State'][j][i]))

                    # Heat Release Rate
                    if check_flags['Flame Processing'].get('Heat Release Rate Cantera', False):
                        outfile.write(" {0:<55e}".format(
                            collective_results['Flame']['Smooth']['Max Heat Release Rate Cantera'][i]))

                    if check_flags['Flame Processing'].get('Heat Release Rate PeleC', False):
                        outfile.write(
                            " {0:<55e}".format(collective_results['Flame']['Smooth']['Max Heat Release Rate PeleC'][i]))

                    # Flame Thickness
                    if check_flags['Flame Processing'].get('Flame Thickness', False):
                        outfile.write(" {0:<55e}".format(collective_results['Flame']['Smooth']['Flame Thickness'][i]))

                    # Surface Length
                    if check_flags['Flame Processing'].get('Surface Length', False):
                        outfile.write(" {0:<55e}".format(collective_results['Flame']['Smooth']['Surface Length'][i]))
                    outfile.write("\n")
            outfile.close()
            return

    """

    """
    # Step 1:
    print('Starting Raw Data Loading and Processing')
    plt_result = parallel_processing_function(plt_dir, (check_flags, animation_bnd_arr, domain_info, output_dir,),
                                              single_pltfile_processing, n_proc)
    print('Completed Raw Data Loading and Processing')

    # Step 2:
    collective_results = {}
    for master_key in plt_result[0]:
        if master_key not in collective_results:
            collective_results[master_key] = {}
        for slave_key in plt_result[0][master_key]:
            # Step 2.1
            temp_vec = []
            for i in range(len(plt_result)):
                temp_vec.append(plt_result[i][master_key][slave_key])
            # Step 2.2
            collective_results[master_key][slave_key] = temp_vec

    # Step 3:
    if 'Flame Processing' in check_flags:
        print('Starting Flame Processing')
        # Step 3.1: Velocity
        if check_flags['Flame Processing'].get('Velocity', False):
            collective_results['Flame']['Velocity'] = np.gradient(
                collective_results['Flame']['Position']) / np.gradient(collective_results['Time']['Value'])
        # Step 3.2: Relative Velocity
        if check_flags['Flame Processing'].get('Relative Velocity', False):
            if check_flags['Flame Processing'].get('Velocity', False):
                collective_results['Flame']['Relative Velocity'] = collective_results['Flame']['Velocity'] - \
                                                                   collective_results['Flame']['Gas Velocity']
            else:
                print('ERROR: Must Enable Velocity Flag to compute the relative velocity')

        # Step 3.3: Data Smoothing
        if check_flags['Flame Processing'].get('Smoothing', False) and check_flags['Flame Processing'].get('Position',
                                                                                                           False):
            smoothing_function(collective_results, 'Flame', 'Position')

        if check_flags['Flame Processing'].get('Smoothing', False) and check_flags['Flame Processing'].get('Velocity',
                                                                                                           False):
            smoothing_function(collective_results, 'Flame', 'Velocity')

        if check_flags['Flame Processing'].get('Smoothing', False) and check_flags['Flame Processing'].get(
                'Relative Velocity', False):
            smoothing_function(collective_results, 'Flame', 'Relative Velocity')

        if check_flags['Flame Processing'].get('Smoothing', False) and check_flags['Flame Processing'].get(
                'Thermodynamic State', False):
            smoothing_function(collective_results, 'Flame', 'Thermodynamic State')

        if check_flags['Flame Processing'].get('Smoothing', False) and (
        check_flags['Flame Processing'].get('Heat Release Rate Cantera', False)):
            smoothing_function(collective_results, 'Flame', 'Max Heat Release Rate Cantera')

        if check_flags['Flame Processing'].get('Smoothing', False) and (
        check_flags['Flame Processing'].get('Heat Release Rate PeleC', False)):
            smoothing_function(collective_results, 'Flame', 'Max Heat Release Rate PeleC')

        if check_flags['Flame Processing'].get('Smoothing', False) and check_flags['Flame Processing'].get(
                'Flame Thickness', False):
            smoothing_function(collective_results, 'Flame', 'Flame Thickness')

        if check_flags['Flame Processing'].get('Smoothing', False) and check_flags['Flame Processing'].get(
                'Surface Length', False):
            smoothing_function(collective_results, 'Flame', 'Surface Length')

        print('Completed Flame Processing')
    # Step 4: Lead Shock Processing
    if 'Leading Shock Processing' in check_flags:
        print('Starting Lead Shock Processing')
        # Step 4.1: Velocity
        if check_flags['Leading Shock Processing'].get('Velocity', False):
            collective_results['Lead Shock']['Velocity'] = np.gradient(
                collective_results['Lead Shock']['Position']) / np.gradient(collective_results['Time']['Value'])
        # Step 4.1: Data Smoothing
        if check_flags['Leading Shock Processing'].get('Smoothing', False) and check_flags[
            'Leading Shock Processing'].get('Position', False):
            smoothing_function(collective_results, 'Lead Shock', 'Position')

        if check_flags['Leading Shock Processing'].get('Smoothing', False) and check_flags[
            'Leading Shock Processing'].get('Velocity', False):
            smoothing_function(collective_results, 'Lead Shock', 'Velocity')

        print('Completed Lead Shock Processing')
    # Step 5: Maximum Pressure Processing
    if 'Maximum Pressure Processing' in check_flags:
        print('Starting Max Pressure Processing')
        # Step 5.1: Data Smoothing
        if check_flags['Maximum Pressure Processing'].get('Smoothing', False) and check_flags[
            'Maximum Pressure Processing'].get('Position', False):
            smoothing_function(collective_results, 'Maximum Pressure', 'Position')

        if check_flags['Maximum Pressure Processing'].get('Smoothing', False) and check_flags[
            'Maximum Pressure Processing'].get('Thermodynamic State', False):
            smoothing_function(collective_results, 'Maximum Pressure', 'Thermodynamic State')

        print('Completed Max Pressure Processing')
    # Step 6: Pre-Shock Processing
    if 'Pre-Shock Processing' in check_flags:
        print('Starting Pre-Shock Processing')
        # Step 6.1: Data Smoothing
        if check_flags['Pre-Shock Processing'].get('Smoothing', False) and check_flags['Pre-Shock Processing'].get(
                'Thermodynamic State', False):
            smoothing_function(collective_results, 'Pre-Shock', 'Thermodynamic State')

        print('Completed Pre-Shock Processing')
    # Step 7: Post-Shock Processing
    if 'Post-Shock Processing' in check_flags:
        print('Starting Post-Shock Processing')
        # Step 7.1: Data Smoothing
        if check_flags['Post-Shock Processing'].get('Smoothing', False) and check_flags['Post-Shock Processing'].get(
                'Thermodynamic State', False):
            smoothing_function(collective_results, 'Post-Shock', 'Thermodynamic State')

        print('Completed Post-Shock Processing')

    # Step 4: Write to file, if any of the sub-dictionary values except 'Domain State Animations' are true
    print('Start Output File Writing')
    for key, sub_dict in check_flags.items():
        if 'Smoothing' in sub_dict and sub_dict['Smoothing'] == True:
            smoothing_flag = True
        else:
            smoothing_flag = False
        if key != 'Domain State Animations':
            if any(sub_dict.values()):  # If any value is True
                write_to_file = True
                break  # Break as soon as we find a True value

    if write_to_file:
        file_output(ensure_long_path_prefix(os.path.join(output_dir, 'Wave-Tracking-Results.txt')), False)
        if smoothing_flag:
            file_output(ensure_long_path_prefix(os.path.join(output_dir, 'Wave-Tracking-Smooth-Results.txt')), True)

        for master_key, sub_dict in check_flags.items():
            if 'Smoothing' in sub_dict and sub_dict['Smoothing'] == True:
                file_output(ensure_long_path_prefix(os.path.join(output_dir, 'Wave-Tracking-Smooth-Results.txt')),
                            True)
    print('Completed Output File Writing')

    # Step 7: Create Variable Evolution
    print('Starting Animation Processing')
    if 'Domain State Animations' in check_flags:
        plot_check = check_flags.get('Domain State Animations', {})

        for key, value in plot_check.items():
            if isinstance(value, (np.ndarray, list, tuple)):
                bool_check = value[0]
                pelec_key = value[1]
            else:
                bool_check = value
                pelec_key = None

            if bool_check:
                # Create directory for plt files
                temp_plt_dir = ensure_long_path_prefix(
                    os.path.join(output_dir, f"Animation-Frames", f"{key}-Plt-Files"))

                state_animation(
                    method='Animate',
                    folder_path=temp_plt_dir,
                    animation_filename=ensure_long_path_prefix(
                        os.path.join(output_dir, f"{key}-Evolution-Animation.mp4")),
                )

        print('Completed Animation Processing')

    return


def main():
    ####################################################################################################################
    # This code is developed to process a 2D Planar Flame simulated using PeleC for a given y-position and a given
    # (temperature) isotherm for the flame and pressure for any shock
    #
    # All functions are configured for a 2 dimensional space
    ####################################################################################################################
    # Step 0: Set all the desired tasks to be performed bny the python script
    skip_load = 0
    row_index = "Center"  # Desired row location for data collection
    ddt_plt_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Raw-PeleC-Data', 'plt308219')

    check_flag_dict = {
        'Flame Processing': {
            'Position': True,
            'Velocity': True,
            'Relative Velocity': False,
            'Thermodynamic State': False,
            'Heat Release Rate Cantera': False,
            'Heat Release Rate PeleC': False,
            'Flame Thickness': False,
            'Surface Length': False,
            'Smoothing': False
        },
        'Leading Shock Processing': {
            'Position': False,
            'Velocity': False,
            'Smoothing': False
        },
        'Maximum Pressure Processing': {
            'Position': False,
            'Thermodynamic State': False,
            'Smoothing': False
        },
        'Pre-Shock Processing': {
            'Thermodynamic State': False,
            'Smoothing': False
        },
        'Post-Shock Processing': {
            'Thermodynamic State': False,
            'Smoothing': False
        },
        'Domain State Animations': {
            'Temperature': (True, 'Temp'),
            'Pressure': (True, 'pressure'),
            'Velocity': (False, 'x_velocity'),
            'Species': False,
            'Heat Release Rate Cantera': (False, None),
            'Heat Release Rate PeleC': (False, None),
            'Flame Thickness': (False, None),
            'Combined': (('Temperature', 'Pressure'), ('Temp', 'pressure'))
        },
        'Local State Animations': {
            'Temperature': True,
            'Pressure': False,
            'Velocity': False,
            'Species': False,
            'Heat Release Rate Cantera': False,
            'Heat Release Rate PeleC': False,
            'Flame Thickness': False,
            # 'Combined': ('Temp', 'pressure')
        }
    }

    # Step 1: Initialize the code with the desired processed variables and mixture composition
    initialize_parameters(
        T=503.15,
        P=10.0 * 100000,
        Phi=1.0,
        Fuel='H2',
        mech='Li-Dryer-H2-mechanism.yaml',
    )

    # Step 3: Collect all the present pelec data directories
    dir_path = os.path.dirname(os.path.realpath(__file__))

    time_data_dir = []
    for raw_data_folder in os.listdir(dir_path):
        if os.path.isdir(os.path.join(dir_path, raw_data_folder)) and raw_data_folder.startswith('Raw-PeleC'):
            raw_data_dir = os.path.join(dir_path, raw_data_folder)
            for time_step in os.listdir(raw_data_dir):
                if os.path.isdir(os.path.join(raw_data_dir, time_step)) and time_step.startswith('plt'):
                    time_data_dir.append(os.path.join(raw_data_dir, time_step))

    # Step 4: Chronologically order the pltFiles and truncate the raw data list if skip loading is enabled
    updated_data_list = sort_files(time_data_dir)

    if skip_load > 0:
        updated_data_list = updated_data_list[0::skip_load]

    # Step 5: Determine the domain sizing parameters (size, # of cells)
    if row_index == 'DDT':
        domain_info = domain_size_parameters(ddt_plt_dir, row_index)
    else:
        domain_info = domain_size_parameters(updated_data_list[0], row_index)

    # Step 6: Create the result directories
    if os.path.exists(os.path.join(dir_path, f"Processed-Global-Results")) is False:
        os.mkdir(os.path.join(dir_path, f"Processed-Global-Results"))
    output_dir_path = os.path.join(dir_path, f"Processed-Global-Results", f"y-{domain_info[1][0][1]:.3g}cm")

    # Step 7: Pre-determine the maximum value present in simulations using the DDT "frame" as the center of a bin
    ddt_idx = updated_data_list.index(ddt_plt_dir)
    temp_plt_files = updated_data_list[
                     max(0, ddt_idx - plotting_bnds_bin):min(len(updated_data_list), ddt_idx + plotting_bnds_bin + 1)]
    temp_max_arr = parallel_processing_function(temp_plt_files, (domain_info, check_flag_dict,), animation_bounds,
                                                n_proc)

    # Find the number of rows and columns (assuming all sub-arrays have the same structure)
    n_rows = len(temp_max_arr[0])  # Number of rows (same for all sub-arrays)
    n_cols = len(temp_max_arr)  # Number of sub-arrays
    # Initialize an empty list to store final results
    final_results = []
    # Iterate over each row
    for i in range(n_rows):
        row_values_2nd_col = []
        row_values_3rd_col = []
        row_texts = []
        # Extract the text and values from the 2nd and 3rd columns
        for j in range(n_cols):
            text = temp_max_arr[j][i][0]  # Extract the 'text' part
            value_2nd_col = temp_max_arr[j][i][1]  # Extract the value from the 2nd column
            value_3rd_col = temp_max_arr[j][i][2]  # Extract the value from the 3rd column

            row_values_2nd_col.append(value_2nd_col)
            row_values_3rd_col.append(value_3rd_col)
            row_texts.append(text)
        # Find the index of the min value in the 2nd column and max value in the 3rd column
        min_index_2nd_col = row_values_2nd_col.index(min(row_values_2nd_col))
        max_index_3rd_col = row_values_3rd_col.index(max(row_values_3rd_col))
        # Retrieve the text, min value, and max value
        min_text = row_texts[min_index_2nd_col]
        min_value = row_values_2nd_col[min_index_2nd_col]
        max_value = row_values_3rd_col[max_index_3rd_col]
        # Append the final result as a list of [text, min value, max value]
        final_results.append([min_text, min_value, max_value])

    # Convert the result to a numpy array (optional)
    animation_bnd_arr = np.array(final_results, dtype=object)

    # Step 8: Individual PltFile Processing
    pelec_processing_function(updated_data_list, domain_info, animation_bnd_arr, check_flag_dict, output_dir_path)

    return


if __name__ == '__main__':
    main()