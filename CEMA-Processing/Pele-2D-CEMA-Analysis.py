import os, yt, multiprocessing, re, itertools
from scipy.interpolate import RegularGridInterpolator, LinearNDInterpolator, griddata
from scipy.linalg import eig, pinv
from matplotlib.tri import Triangulation
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import cantera as ct
import numpy as np
from geomdl import BSpline
from geomdl import utilities

########################################################################################################################
# Global Variables
########################################################################################################################

n_procs = 1

data_set = 'Test'

########################################################################################################################
# Global Classes
########################################################################################################################

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
    input_params.update_composition()   # Update oxygen amount and composition
    input_params.load_mechanism_species()  # Load species from the mechanism file

########################################################################################################################
# Function Scripts - Data Handeling
########################################################################################################################

def worker_function(args):
    """Worker function to process data and return the result."""
    iter_var, const_list, shared_input_params, predicate, kwargs = args
    global input_params
    input_params = shared_input_params
    result = predicate((iter_var, const_list, kwargs))
    return result

def parallel_processing_function(iter_arr, const_list, predicate, **kwargs):
    if n_procs > 1:
        """Perform parallel processing using multiprocessing.Pool."""
        print('Initializing Parallel Processing with ', n_procs, ' cores')
        # Perform the multiprocessing
        with multiprocessing.Pool(
                processes=n_procs, initializer=init_pool, initargs=(input_params,)
        ) as pool:
            # Use itertools.repeat for constant arguments
            tasks = zip(
                iter_arr,
                itertools.repeat(const_list),
                itertools.repeat(input_params),
                itertools.repeat(predicate),
                itertools.repeat(kwargs)
            )

            # Perform parallel mapping
            results = pool.map(worker_function, tasks)

    else:
        """Run the function in without parallel."""
        print("Performing Calculations in Serial")
        results = []
        for i in range(len(iter_arr)):
            results.append(worker_function((iter_arr[i], const_list, input_params, predicate, kwargs)))

    return results

def init_pool(global_params):
    """Initializer function to set the global variable."""
    global input_params
    input_params = global_params

########################################################################################################################
# Function Scripts - External Data Handling
########################################################################################################################

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
        file_list: A list of full paths to plot folders to be sorted

    Returns:
        sorted_list: A list of sorted file paths
    """
    def extract_number(file_path):
        # Get the base folder name from the full path
        folder_name = os.path.basename(file_path)
        # Extract the numeric part following "plt" in the folder name
        match = re.search(r'plt(\d+)', folder_name)
        return int(match.group(1)) if match else float('inf')

    return sorted(file_list, key=extract_number)

########################################################################################################################
# Function Scripts - Data Generation (Cantera)
########################################################################################################################

def cantera_flame():

    return

def sdtoolbox_detonation():

    return

########################################################################################################################
# Function Scripts - DNS (Pele) Data Processing
########################################################################################################################
def pelec_flame_data(args):

    global input_params
    iter_var, const_arr, kwargs = args

    pltFile_dir = iter_var
    x_bnds, y_bnds = const_arr

    # Step 1: Load the pelec data
    raw_data = yt.load(pltFile_dir)

    # Step 2: Get the maximum refinement level and smallest grid spacing
    max_level = raw_data.index.max_level
    dx = raw_data.index.get_smallest_dx().to_value()

    # Step 3: Extract the desired refinement level grid
    grids = [grid for grid in raw_data.index.grids if grid.Level == max_level]

    # Step 4: Extract all the unique y-values present in the desired refinement level grid
    x_values, y_values = [], []
    for grid in grids:
        x_values.extend(grid['boxlib', 'x'].to_value().flatten())
        y_values.extend(grid['boxlib', 'y'].to_value().flatten())

    # Get unique y values and store them in an array
    x_arr = np.unique(x_values)
    y_arr = np.unique(y_values)

    # Update the x bounds to be a valid grid point
    x_min = x_arr[np.argmin(np.abs(x_arr - x_bnds[0]))]
    x_max = x_arr[np.argmin(np.abs(x_arr - x_bnds[1]))]
    x_arr = x_arr[(x_arr >= x_min) & (x_arr <= x_max)]

    y_min = y_arr[np.argmin(np.abs(y_arr - y_bnds[0]))]
    y_max = y_arr[np.argmin(np.abs(y_arr - y_bnds[1]))]
    y_arr = y_arr[(y_arr >= y_min) & (y_arr <= y_max)]

    # Set the number of points present between the min and max values
    x_pts = round((x_max - x_min) / dx) + 1
    y_pts = round((y_max - y_min) / dx) + 1

    # Create mapping of physical coordinates to array indices
    x_index_map = {x: i for i, x in enumerate(x_arr)}
    y_index_map = {y: i for i, y in enumerate(y_arr)}

    # Step 4: Create an empty dictionary for each variable and place an empty object array with length equal to the
    # number of y-values

    target_strings = ["Temp", "pressure", "density", "x_velocity", "y_velocity", "rho_", "Y(", "D(", "conductivity", "cp", "heatRelease", 'viscosity']

    data_dict = {
        var: np.full((y_pts, x_pts), np.nan)  # Initialize with NaNs
        for _, var in raw_data.field_list if any(var.startswith(prefix) for prefix in target_strings)
    }

    data_dict['x'] = np.full((y_pts, x_pts), np.nan)  # Initialize with NaNs
    data_dict['y'] = np.full((y_pts, x_pts), np.nan)  # Initialize with NaNs
    data_dict['grid'] = np.full((y_pts, x_pts), np.nan, dtype=object)  # Initialize with NaNs
    data_dict['pltFile'] = pltFile_dir

    # Step 5:
    for grid in grids:
        x_vals = grid['boxlib', 'x'].to_value().flatten()
        y_vals = grid['boxlib', 'y'].to_value().flatten()

        for i in range(len(x_vals)):
            x, y = x_vals[i], y_vals[i]
            if x in x_index_map and y in y_index_map:  # Ensure valid points
                xi, yi = x_index_map[x], y_index_map[y]  # Get array indices
                data_dict['x'][yi, xi] = x
                data_dict['y'][yi, xi] = y
                data_dict['grid'][yi, xi] = (x, y)
                for var in data_dict.keys():
                    if var not in ('grid', 'pltFile'):
                        data_dict[var][yi, xi] = grid["boxlib", var].to_value().flatten()[i]  # Place value at the correct index

    return data_dict

def flame_contour(data_info, y_loc=None, box_bnds=False, box_size=None, tracking_str='Temp', tracking_val=2000):
    def manually_aquire_flame_contour():
        # Get the maximum refinement level
        max_level = raw_data.index.max_level
        # Initialize containers for the highest level data
        x_coords, y_coords, tracking_arr = [], [], []
        # Loop through grids and extract data at the highest level
        for grid in raw_data.index.grids:
            if grid.Level == max_level:
                x_coords.append(grid["boxlib", "x"].to_value().flatten())
                y_coords.append(grid["boxlib", "y"].to_value().flatten())
                tracking_arr.append(grid[tracking_str].flatten())

        x_coords = np.concatenate(x_coords)
        y_coords = np.concatenate(y_coords)
        tracking_arr = np.concatenate(tracking_arr)

        # Create a triangulation
        triangulation = Triangulation(x_coords, y_coords)

        # Use tricontour to compute the contour line
        contour = plt.tricontour(triangulation, tracking_arr, levels=[tracking_val])

        # If no contour is found, use grid interpolation
        if not contour.collections:
            print("No contour found at the specified level. Using interpolation...")

            # Create a regular grid for interpolation
            xi = np.linspace(np.min(x_coords), np.max(x_coords), 1e4)
            yi = np.linspace(np.min(y_coords), np.max(y_coords), 1e4)
            xi, yi = np.meshgrid(xi, yi)

            # Interpolate temperatures onto the regular grid
            tracking_grid = griddata((x_coords, y_coords), tracking_arr, (xi, yi), method='cubic')

            # Create a new triangulation on the regular grid and compute the contour
            triangulation = Triangulation(xi.flatten(), yi.flatten())
            contour = plt.tricontour(triangulation, tracking_grid.flatten(), levels=[tracking_val])

        # Extract the contour line vertices
        paths = contour.collections[0].get_paths()
        contour_points = np.vstack([path.vertices for path in paths])

        return contour_points

    ###########################################
    # Main Function
    ###########################################

    # Step 1: Load the pelec data
    if isinstance(data_info, str):
        raw_data = yt.load(data_info)
    else:
        raw_data = yt.load(data_info['pltFile'])

    # Step 2: Determine the flame location using triangulation or yt.isocontour
    try:
        flame_vert = manually_aquire_flame_contour()
    except Exception as e:
        raw_data.force_periodicity()
        flame_vert = raw_data.all_data().extract_isocontours(tracking_str, tracking_val)
        print(f"Error: Unable to manually extract flame contour: {e}")

    flame_vert = flame_vert[np.abs(np.mean(flame_vert[:, 0]) - flame_vert[:, 0]) <= 1]

    if box_size:
        # Determine the flame index
        flame_idx = np.argmin(np.abs(flame_vert[:, 1] - y_loc))
        flame_loc = [flame_vert[flame_idx, 0], flame_vert[flame_idx, 1]]

        x_bnds = [flame_loc[0] - box_size, flame_loc[0] + box_size]
        y_bnds = [flame_loc[1] - box_size, flame_loc[1] + box_size]

        # Box flame contour
        mask_x = (flame_vert[:, 0] >= x_bnds[0]) & (flame_vert[:, 0] <= x_bnds[-1])
        mask_y = (flame_vert[:, 1] >= y_bnds[0]) & (flame_vert[:, 1] <= y_bnds[-1])
        mask = mask_x & mask_y

        if box_bnds:
            return x_bnds, y_bnds
        else:
            return flame_vert[mask]

    else:
        if box_bnds:
            return [np.min(flame_vert[:, 0]), np.max(flame_vert[:, 0])], [0, raw_data.domain_right_edge[1].to_value()]
        else:
            return flame_vert

def flame_points(data):
    # Step 1: Extract the flame contour
    contour = flame_contour(data)

    # Step 2: Define bounds
    x_min, x_max = np.nanmin(data['x']), np.nanmax(data['x'])
    y_min, y_max = np.nanmin(data['y']), np.nanmax(data['y'])

    # Step 3: Filter contour points within bounds while maintaining order
    valid_mask = (contour[:, 0] >= x_min) & (contour[:, 0] <= x_max) & \
                 (contour[:, 1] >= y_min) & (contour[:, 1] <= y_max)
    contour = contour[valid_mask]

    # Step 4: Map the filtered contour to grid points
    x_arr = np.unique(data['x'])[~np.isnan(np.unique(data['x']))]
    y_arr = np.unique(data['y'])[~np.isnan(np.unique(data['y']))]

    flame_pts = []
    for x, y in contour:
        x_idx = np.argmin(np.abs(x_arr - x))
        y_idx = np.argmin(np.abs(y_arr - y))
        flame_pts.append((x_idx, y_idx))  # Keep as list to preserve order

    return contour, np.array(flame_pts)

def flame_normal(contour):
    def fit_curve(contour, degree=3):
        """
        Fit a NURBS curve to the given contour points.

        Parameters:
            contour (ndarray): Nx2 array of (x, y) points.
            degree (int): Degree of the NURBS curve.

        Returns:
            curve (BSpline.Curve): Fitted NURBS curve object.
        """
        num_ctrl_pts = len(contour)  # Number of control points

        # Create a B-Spline curve
        curve = BSpline.Curve()
        curve.degree = min(degree, num_ctrl_pts - 1)  # Ensure degree is valid
        curve.ctrlpts = contour.tolist()  # Set control points
        curve.knotvector = utilities.generate_knot_vector(curve.degree, num_ctrl_pts)
        curve.delta = 0.01  # Controls evaluation resolution

        return curve

    def compute_normals(curve, num_samples=100):
        """
        Compute normal vectors along the NURBS curve.

        Parameters:
            curve (BSpline.Curve): Fitted NURBS curve.
            num_samples (int): Number of points to sample along the curve.

        Returns:
            points (ndarray): Nx2 array of evaluated curve points.
            normals (ndarray): Nx2 array of unit normal vectors.
        """
        u_vals = np.linspace(0, 1, num_samples)  # Parameter values
        points = np.array([curve.evaluate_single(u) for u in u_vals])

        # Compute first derivative (tangent vectors)
        tangents = np.array([curve.derivatives(u, order=1)[1] for u in u_vals])

        # Compute normal vectors (rotate tangents by 90 degrees)
        normals = np.column_stack((tangents[:, 1], -tangents[:, 0]))

        # Normalize normal vectors
        norms = np.linalg.norm(normals, axis=1)
        normals = normals / norms[:, np.newaxis]

        return points, normals

    # Fit the curve
    curve = fit_curve(contour, degree=3)

    # Compute normals
    # points, normals = compute_normals(curve, num_samples=len(contour[:, 0]))
    points, normals = compute_normals(curve, num_samples=10)

    plt_flag = True
    if plt_flag:
        plt.figure(figsize=(8, 6))
        plt.plot(contour[:, 0], contour[:, 1], 'bo-', label="Original Contour")
        plt.plot(points[:, 0], points[:, 1], 'r-', label="NURBS Curve")
        plt.quiver(points[:, 0], points[:, 1], normals[:, 0], normals[:, 1], color='g', scale=20, label="Normals")
        plt.legend()
        plt.axis("equal")
        plt.show()

    return points, normals


def flame_normal_val(data, normal_pts, normal_arr, var, marker_val = None, PLT_FLAG=False):
    def normal_vector(normal_vec, x_arr, y_arr):
        # Step 1: Determine the spacing to be used for the normal vector
        dx = np.abs(np.unique(x_arr)[1] - np.unique(x_arr)[0])
        dy = np.abs(np.unique(y_arr)[1] - np.unique(y_arr)[0])
        t_step = min(dx, dy) / np.linalg.norm(normal_vec) / 1e0  # Adjust step size for resolution

        x_val = normal_pts[0]
        y_val = normal_pts[1]

        # Step 2: Generate points forward along the normal vector
        valid_line_points_forward = []
        t = 0
        while True:
            # Calculate the new point in the forward direction
            x_new = x_val + t * normal_vec[0]
            y_new = y_val + t * normal_vec[1]

            # Check bounds
            if (x_new < np.min(data['x'])) or (x_new > np.max(data['x'])) or (y_new < np.min(data['y'])) or (
                    y_new > np.max(data['y'])):
                break

            # Append new point if valid
            x_idx = np.searchsorted(np.unique(data['x']), x_new)
            y_idx = np.searchsorted(np.unique(data['y']), y_new)

            if (0 <= x_idx < data['x'].shape[1] and 0 <= y_idx < data['y'].shape[0] and
                    not np.isnan(data['x'][y_idx, x_idx]) and not np.isnan(data['y'][y_idx, x_idx])):
                valid_line_points_forward.append([x_new, y_new])
            else:
                break  # Stop if we hit a NaN

            t += t_step  # Move forward

        # Step 3: Generate points backward along the normal vector
        valid_line_points_backward = []
        t = 0
        while True:
            # Calculate the new point in the backward direction
            x_new = x_val - t * normal_vec[0]
            y_new = y_val - t * normal_vec[1]

            # Check bounds
            if (x_new < np.min(data['x'])) or (x_new > np.max(data['x'])) or (y_new < np.min(data['y'])) or (
                    y_new > np.max(data['y'])):
                break

            # Append new point if valid
            x_idx = np.searchsorted(np.unique(data['x']), x_new)
            y_idx = np.searchsorted(np.unique(data['y']), y_new)

            if (0 <= x_idx < data['x'].shape[1] and 0 <= y_idx < data['y'].shape[0] and
                    not np.isnan(data['x'][y_idx, x_idx]) and not np.isnan(data['y'][y_idx, x_idx])):
                valid_line_points_backward.append([x_new, y_new])
            else:
                break  # Stop if we hit a NaN

            t += t_step  # Move backward

        # Combine forward and backward points
        valid_line_points = valid_line_points_backward[::-1] + valid_line_points_forward  # Reverse backward points

        return np.array(valid_line_points)

    ###########################################
    # Main Function
    ###########################################

    # Step 1: Determine the normal line
    normal_line = normal_vector(normal_arr, data['x'], data['y'])

    # Step 2: Define bounds for valid mask
    x_min, x_max = np.min(normal_line[:, 0]), np.max(normal_line[:, 0])
    y_min, y_max = np.min(normal_line[:, 1]), np.max(normal_line[:, 1])

    # Step 3: Apply bounding box mask
    valid_mask = ~np.isnan(var) & \
                 (data['x'] >= x_min) & (data['x'] <= x_max) & \
                 (data['y'] >= y_min) & (data['y'] <= y_max)

    valid_x_pts = data['x'][valid_mask]
    valid_y_pts = data['y'][valid_mask]

    # Step 4: Create normal mask based on valid points
    valid_x_min, valid_x_max = valid_x_pts.min(), valid_x_pts.max()
    valid_y_min, valid_y_max = valid_y_pts.min(), valid_y_pts.max()

    normal_mask = (normal_line[:, 0] >= valid_x_min) & (normal_line[:, 0] <= valid_x_max) & \
                  (normal_line[:, 1] >= valid_y_min) & (normal_line[:, 1] <= valid_y_max)

    # Step 5: Apply the normal mask to the normal line
    normal_line = normal_line[normal_mask]

    # Step 6: Recalculate valid mask based on the masked normal line
    if normal_line.size > 0:  # Check if there are valid points in masked normal line
        x_min_masked, x_max_masked = np.min(normal_line[:, 0]), np.max(normal_line[:, 0])
        y_min_masked, y_max_masked = np.min(normal_line[:, 1]), np.max(normal_line[:, 1])

        valid_mask = valid_mask & \
                     (data['x'] >= x_min_masked) & (data['x'] <= x_max_masked) & \
                     (data['y'] >= y_min_masked) & (data['y'] <= y_max_masked)

    # Step 7: Interpolation restricted to valid mask
    adjusted_normal_line = normal_line[:, [1, 0]]  # Swap columns for adjusted line
    """
    try:
        points = np.column_stack((data['x'][valid_mask], data['y'][valid_mask]))
        values = var[valid_mask].reshape(len(np.unique(points[:, 1])), len(np.unique(points[:, 0])))
        interpolator = RegularGridInterpolator((np.unique(points[:, 1]), np.unique(points[:, 0])), values,
                                               bounds_error=False)
        interp_mask = ~np.isnan(interpolator(adjusted_normal_line))

        if PLT_FLAG:
            plt.figure(figsize=(8, 6))
            # Use pcolormesh for uniform grids
            mesh = plt.pcolormesh(np.unique(points[:, 0]), np.unique(points[:, 1]),
                                  np.array(values).reshape(np.unique(points[:, 1]).shape[0],
                                                           np.unique(points[:, 0]).shape[0]), shading='auto',
                                  cmap='hot')
            # Scatter plot for the normal line
            plt.scatter(normal_line[:, 0], normal_line[:, 1], c=interpolator(adjusted_normal_line), cmap='hot')
            cbar = plt.colorbar(mesh)
            cbar.set_label("Temperature [K]")
            plt.title("Flame Normal and Temperature Interpolation")
            plt.show()
    except:
    """
    points = np.column_stack((data['y'][valid_mask], data['x'][valid_mask]))
    values = var[valid_mask]
    interpolator = lambda xi: griddata(points, var[valid_mask], xi, method='cubic')
    interp_mask = ~np.isnan(interpolator(adjusted_normal_line))

    if PLT_FLAG:
        plt.figure(figsize=(8, 6))
        # Use tricontourf for non-uniform grids
        mesh = plt.tricontourf(points[:, 1], points[:, 0], values, levels=100, cmap='hot')
        # Scatter plot for the normal line
        plt.scatter(normal_line[:, 0], normal_line[:, 1],
                    c=interpolator(adjusted_normal_line), cmap='hot', edgecolor='k')
        cbar = plt.colorbar(mesh)
        cbar.set_label("Temperature [K]")
        plt.title("Flame Normal and Temperature Interpolation")
        plt.xlabel("X-axis label")  # Add your x-axis label
        plt.ylabel("Y-axis label")  # Add your y-axis label
        plt.ylim(np.nanmin(points[:, 0]), np.nanmax(points[:, 0]))
        plt.xlim(np.nanmin(points[:, 1]), np.nanmax(points[:, 1]))
        plt.show()

    if marker_val:
        return interpolator(adjusted_normal_line)[interp_mask], np.argmin(abs(interpolator(adjusted_normal_line)[interp_mask] - marker_val))
    else:
        return interpolator(adjusted_normal_line)[interp_mask]

########################################################################################################################
# Function Scripts - General Mathematics Processing
########################################################################################################################

def spatial_derivative(arr, dx, direction):
    ###########################################
    # Internal Functions
    ###########################################

    def derivative(dimensionality, i=None, j=None):

        if dimensionality == '1D':
            if 1 <= i < x_pts - 1 and not np.isnan(arr[i - 1]) and not np.isnan(arr[i + 1]):
                return (arr[i + 1] - arr[i - 1]) / (2 * dx)

                # Forward Difference (Second Order)
            elif i < x_pts - 2 and not np.isnan(arr[j, i + 1]) and not np.isnan(arr[j, i + 2]):
               return (-3 * arr[i] + 4 * arr[i + 1] - arr[i + 2]) / (2 * dx)

                # Backward Difference (Second Order)
            elif i > 1 and not np.isnan(arr[j, i - 1]) and not np.isnan(arr[j, i - 2]):
                return (3 * arr[i] - 4 * arr[i - 1] + arr[i - 2]) / (2 * dx)

        elif dimensionality == '2D-x':
            if 1 <= i < x_pts - 1 and not np.isnan(arr[j, i - 1]) and not np.isnan(arr[j, i + 1]):
                return (arr[j, i + 1] - arr[j, i - 1]) / (2 * dx)

            # Forward Difference (Second Order)
            elif i < x_pts - 2 and not np.isnan(arr[j, i + 1]) and not np.isnan(arr[j, i + 2]):
                return (-3 * arr[j, i] + 4 * arr[j, i + 1] - arr[j, i + 2]) / (2 * dx)

            # Backward Difference (Second Order)
            elif i > 1 and not np.isnan(arr[j, i - 1]) and not np.isnan(arr[j, i - 2]):
                return (3 * arr[j, i] - 4 * arr[j, i - 1] + arr[j, i - 2]) / (2 * dx)

        elif dimensionality == '2D-y':
            # Centered Difference (Second Order)
            if 1 <= j < y_pts - 1 and not np.isnan(arr[j - 1, i]) and not np.isnan(arr[j + 1, i]):
                return (arr[j + 1, i] - arr[j - 1, i]) / (2 * dx)

            # Forward Difference (Second Order)
            elif j < y_pts - 2 and not np.isnan(arr[j + 1, i]) and not np.isnan(arr[j + 2, i]):
                return (-3 * arr[j, i] + 4 * arr[j + 1, i] - arr[j + 2, i]) / (2 * dx)

            # Backward Difference (Second Order)
            elif j > 1 and not np.isnan(arr[j - 1, i]) and not np.isnan(arr[j - 2, i]):
                return (3 * arr[j, i] - 4 * arr[j - 1, i] + arr[j - 2, i]) / (2 * dx)

        elif dimensionality == '3D':
            raise ValueError("W.I.P. - Please choose another dimensionality")

        else:
            raise ValueError("Unsupported array dimensionality")

        return

    ###########################################
    # Main Function
    ###########################################

    # Step 1: Allocate memory to store the derivative
    dfdx = np.full_like(arr, np.nan)

    # Step 2: Extract the shape of arr
    pts_idx_arr = [(x, y) for x in range(arr.shape[1]) for y in range(arr.shape[0])]

    if arr.ndim == 1:
        x_pts = arr.shape[0]
        for x_idx in pts_idx_arr:
            dfdx[x_idx] = derivative('1D', i=x_idx)

    elif arr.ndim == 2:
        x_pts = arr.shape[1]
        y_pts = arr.shape[0]
        for y_idx, x_idx in pts_idx_arr:
            dfdx[y_idx, x_idx] = derivative(f'2D-{direction}', i=x_idx, j=y_idx)

    else:
        raise ValueError("Unsupported array dimensionality")

    return dfdx

def temporal_derivative():

    return

def steady_state(arr):
    ###########################################
    # Main Function
    ###########################################
    # Step 1:
    dfdt = np.full_like(arr, np.nan)

    # Step 2: Determine the length of sub-arrays
    array_length = None
    for row in arr:
        for cell in row:
            if isinstance(cell, np.ndarray):  # Find first valid sub-array
                array_length = len(cell)
                break
        if array_length is not None:
            break

    # Step32: Extract the shape of arr
    pts_idx_arr = np.indices(arr.shape)

    # Step 4:
    if arr.ndim == 1:
        for x_idx in pts_idx_arr:
            dfdt[x_idx] = np.zeros(array_length)

    elif arr.ndim == 2:
        for y_idx, x_idx in pts_idx_arr:
            dfdt[y_idx, x_idx] = np.zeros(array_length)

    else:
        raise ValueError("Unsupported array dimensionality")

    return dfdt

def total_derivative(dfdt, dfdx=None, dfdy=None, x_vel=None, y_vel=None):
    ###########################################
    # Main Function
    ###########################################
    # Step 1: Allocate memory to store the derivative
    DfDt = np.full_like(dfdt, np.nan)

    # Step 2: Determine the length of sub-arrays
    array_length = None
    for row in dfdt:
        for cell in row:
            if isinstance(cell, np.ndarray):  # Find first valid sub-array
                array_length = len(cell)
                break
        if array_length is not None:
            break

    # Step32: Extract the shape of arr
    pts_idx_arr = np.indices(dfdt.shape)

    # Step 4:
    if dfdt.ndim == 1:
        for x_idx in pts_idx_arr:
            for idx in range(array_length):
                DfDt[x_idx] = dfdt[x_idx][idx] + x_vel[x_idx] * dfdx[x_idx][idx]

    elif dfdt.ndim == 2:
        for y_idx, x_idx in pts_idx_arr:
            for idx in range(array_length):
                DfDt[y_idx, x_idx] = (dfdt[y_idx, x_idx][idx] +
                                      x_vel[y_idx, x_idx] * dfdx[y_idx, x_idx][idx] +
                                      y_vel[y_idx, x_idx] * dfdy[y_idx, x_idx][idx])

    else:
        raise ValueError("Unsupported array dimensionality")

    return DfDt

########################################################################################################################
# Function Scripts - Chemical Explosive Mode Analysis (CEMA) Functions
########################################################################################################################

def cantera_enthalpy(data):

    # Map the temperature and pressure to arrays, and collect the species present in the solution
    temperature = data['Temp']
    pressure = data['pressure']
    species_list = [var for var in input_params.species]
    y_pts, x_pts = data['x'].shape

    # Allocate memory for the array
    species_enthalpy = {
        f'Y({var})': np.full((y_pts, x_pts), np.nan)  # Initialize with NaNs
        for var in species_list
    }

    # Step 2: Create a temporary cantera gas object
    gas_tmp = ct.Solution(input_params.mech)

    for j in range(0, y_pts):
        for i in range(0, x_pts):
            if np.isnan(temperature[j, i]):
                continue  # Skip NaN values
            else:
                for k, species in enumerate(species_list):
                    # Map the species mass fractions to a dict for use with cantera gas object
                    species_dict = {
                        species: data[f"Y({species})"][j, i]  # Construct key dynamically
                        for species in species_list if f"Y({species})" in data  # Ensure key exists
                    }

                    # Set the gas object state
                    gas_tmp.TPY = temperature[j, i], pressure[j, i] / 10, species_dict

                    species_enthalpy[f"Y({species})"][j, i] = gas_tmp.standard_enthalpies_RT[k] * ct.gas_constant * temperature[j,i] * 10000

    return species_enthalpy

########################################################################################################################
# Function Scripts - Chemical Explosive Mode Analysis (CEMA) Functions
########################################################################################################################

def cema_source_terms(data):
    ###########################################
    # Internal Functions
    ###########################################

    def diffusion_terms():
        # Step 1: Pre-calculate the various derivatives
        diffusion_coeffs = [f"D({var})" for var in input_params.species]

        mass_fractions = [f"Y({var})" for var in input_params.species]

        dYdx = dYdy = {
            var: np.full((y_pts, x_pts), np.nan)  # Initialize with NaNs
            for var in mass_fractions
        }

        Jx = Jy = {
            var: np.full((y_pts, x_pts), np.nan)  # Initialize with NaNs
            for var in mass_fractions
        }

        dJdx = dJdy = {
            var: np.full((y_pts, x_pts), np.nan)  # Initialize with NaNs
            for var in mass_fractions
        }

        # Step 2: Calculate the species dependent derivatives (dY, dJ)
        for i, species in enumerate(input_params.species):
            # Species Mass Fraction Derivatives
            dYdx[f'Y({species})'] = spatial_derivative(data[f'Y({species})'], dx, 'x')
            dYdy[f'Y({species})'] = spatial_derivative(data[f'Y({species})'], dy, 'y')
            # Diffusion Flux
            Jx[f'Y({species})'] = data['density'] * data[diffusion_coeffs[i]] * dYdx[f'Y({species})']
            Jy[f'Y({species})'] = data['density'] * data[diffusion_coeffs[i]] * dYdy[f'Y({species})']
            # Diffusion Flux Derivatives
            dJdx[f'Y({species})'] = spatial_derivative(Jx[f'Y({species})'], dx, 'x')
            dJdy[f'Y({species})'] = spatial_derivative(Jy[f'Y({species})'], dy, 'y')

        # Step 3: Calculate the velocity spatial derivatives
        dudx = spatial_derivative(data['x_velocity'], dx, 'x')
        dudy = spatial_derivative(data['x_velocity'], dy, 'y')
        dvdx = spatial_derivative(data['y_velocity'], dx, 'x')
        dvdy = spatial_derivative(data['y_velocity'], dy, 'y')

        # Step 4: Calculate the pressure terms (pressure, shear stress)
        dPdx = spatial_derivative(data['pressure'], dx, 'x')
        dPdy = spatial_derivative(data['pressure'], dy, 'y')

        dPudx = spatial_derivative(data['pressure'] * data['x_velocity'], dx, 'x')
        dPvdy = spatial_derivative(data['pressure'] * data['y_velocity'], dy, 'y')

        tau_xx = data['viscosity'] * (dudx + dudx)
        tau_xy = data['viscosity'] * ((dudy + dvdx) - (2/3) * (dudx + dvdy))
        tau_yx = data['viscosity'] * ((dvdx + dudy) - (2/3) * (dudx + dvdy))
        tau_yy = data['viscosity'] * (dvdy + dvdy)

        # Step 5: Calculate the variables in the source terms (g_d, g_r, g_T)
        # Diffusion Term (g_d)
        g_d = []
        for species in input_params.species:
            g_d.append(dJdx[f'Y({species})'] + dJdy[f'Y({species})'])

        g_d.append(spatial_derivative(tau_xx, dx, 'x') + spatial_derivative(tau_xy, dy, 'y'))
        g_d.append(spatial_derivative(tau_yx, dx, 'x') + spatial_derivative(tau_yy, dy, 'y'))

        species_enthalpy = cantera_enthalpy(data)
        tmp_val_x = data['conductivity'] * spatial_derivative(data['Temp'], dx, 'x') + sum(data['density'] * data[f'D({species})'] * species_enthalpy[f'Y({species})'] * dYdx[f'Y({species})'] for species in input_params.species) + data['x_velocity'] * tau_xx + data['y_velocity'] * tau_yx
        tmp_val_y = data['conductivity'] * spatial_derivative(data['Temp'], dy, 'y') + sum(data['density'] * data[f'D({species})'] * species_enthalpy[f'Y({species})'] * dYdy[f'Y({species})'] for species in input_params.species) + data['x_velocity'] * tau_xy + data['y_velocity'] * tau_yy
        g_d.append(spatial_derivative(tmp_val_x, dx, 'x') + spatial_derivative(tmp_val_y, dy, 'y'))

        # Density Variation Term (g_r)
        g_r = []
        for species in input_params.species:
            g_r.append(-data[f'rho_{species}'] * (dudx + dvdy))

        g_r.append(-data['density'] * data['x_velocity'] * (dudx + dvdy))
        g_r.append(-data['density'] * data['y_velocity'] * (dudx + dvdy))

        g_r.append(-data['rho_e'] * (dudx + dvdy))

        # Pressure and Work Terms (g_T)
        g_T = []
        for species in input_params.species:
            g_T.append(np.zeros_like(data['x']))

        g_T.append(-dPdx)
        g_T.append(-dPdy)

        g_T.append(- (dPudx + dPvdy))

        return g_d, g_r, g_T

    def chemical_terms():
        # Step 1:
        g_w = []
        for species in input_params.species:
            g_w.append(data[f'rho_omega_{species}'] / data['density'])

        g_w.append(data['x'] * 0)
        g_w.append(data['x'] * 0)
        g_w.append(data['x'] * 0)

        return g_w

    ###########################################
    # Main Function
    ###########################################

    # Step 1: Extract grid spacing from domain (constant grid spacing at set level)
    y_pts, x_pts = data['x'].shape
    dx = dy = np.diff(data['x'], axis=1)[~np.isnan(np.diff(data['x'], axis=1))][0]

    # Step 2:
    g_d, g_r, g_T = diffusion_terms()

    # Step 3:
    g_w = chemical_terms()

    return g_w, g_d, g_r, g_T

def cema_jacobian(data, jacobian_numerator):

    # Step 1: Extract the flame points and corresponding normal vectors
    contour, flame_pts = flame_points(data)
    normal_pts, nomral_vecs = flame_normal(contour)

    # Step 2: Calculate the progress variable field based off the H2O (water) mass fraction
    progress_var = (data['Y(H2O)'] - np.min(np.nan_to_num(data['Y(H2O)'], nan=0))) / (np.max(np.nan_to_num(data['Y(H2O)'], nan=0)) - np.min(np.nan_to_num(data['Y(H2O)'], nan=0)))

    # Step 3: Create a dictionary to store the denominator of the jacobian partial derivative
    jacobian_denominator = np.zeros(len(jacobian_numerator), dtype=object)
    for idx, species in enumerate(input_params.species):
        jacobian_denominator[idx] = data['density'] * data[f'Y({species})']
    jacobian_denominator[-3] = data['density'] * data['x_velocity']
    jacobian_denominator[-2] = data['density'] * data['y_velocity']
    jacobian_denominator[-1] = data['rho_e']


    # Step 1: Loop over each point that makes up the flame
    jacobian = np.zeros_like(normal_pts[:, 0], dtype=object)
    for idx, (i, j) in enumerate(normal_pts):
        # Step 1.1: Create a temporary vector to store the progress value along the flame normal
        tmp_prog_val_vec, flame_normal_pt = flame_normal_val(data, normal_pts[idx], nomral_vecs[idx], progress_var, marker_val=0.05, PLT_FLAG=True)
        # Step 1.2: Loop over the components in the jacobian
        tmp_jacobian = np.zeros((len(jacobian_denominator), len(jacobian_numerator)), dtype=object)
        for row in range(len(jacobian_denominator)):
            tmp_denominator = flame_normal_val(data, normal_pts[idx], nomral_vecs[idx], jacobian_denominator[row])
            for column in range(len(jacobian_numerator)):
                tmp_numerator = flame_normal_val(data, normal_pts[idx], nomral_vecs[idx], jacobian_numerator[column])
                tmp_term_1 = np.gradient(tmp_numerator) / np.gradient(tmp_prog_val_vec)
                tmp_term_2 = np.gradient(tmp_prog_val_vec) / np.gradient(tmp_denominator)
                try:
                    tmp_jacobian[row, column] = tmp_term_1[flame_normal_pt] / tmp_term_2[flame_normal_pt]
                except:
                    print('Fail')

        jacobian[idx] = tmp_jacobian

    return normal_pts, jacobian

def cema_mode(data, jacobian, g_w, g_d, g_r, g_T, flame_pts):

    phi_w, phi_f, alpha = [], [], []
    for idx in range(len(jacobian)):
        # Step 1: Extract the eigenvalue and left eigenvector
        eig_val, eig_vec_left, _ = eig(jacobian[idx].astype(float), left=True)

        # Step 2: Determine the mode with the largest positive real part eigenvalue
        valid_mask = eig_val.real > 0
        if (eig_val * valid_mask).size == 0:
            continue

        eig_idx = np.argmax(eig_val * valid_mask)
        lambda_e = (eig_val * valid_mask)[eig_idx]
        b_e = (eig_vec_left * valid_mask)[:, eig_idx]

        # Step 3:
        distances = np.sqrt((data['x'] - flame_pts[idx][0]) ** 2 + (data['y'] - flame_pts[idx][1]) ** 2)
        flame_idx = np.unravel_index(np.nanargmin(distances), distances.shape)
        tmp_g_w, tmp_g_d, tmp_g_r, tmp_g_T = [], [], [], []
        for j in range(len(g_w)):
            tmp_g_w.append(g_w[j][flame_idx[1], flame_idx[0]])
            tmp_g_d.append(g_d[j][flame_idx[1], flame_idx[0]])
            tmp_g_r.append(g_r[j][flame_idx[1], flame_idx[0]])
            tmp_g_T.append(g_T[j][flame_idx[1], flame_idx[0]])

        phi_w.append(np.dot(b_e.real, tmp_g_w))
        phi_f.append(np.dot(b_e.real, np.array(tmp_g_d) + np.array(tmp_g_r) + np.array(tmp_g_T)))
        alpha.append(phi_f[-1] / phi_w[-1])

    return phi_w, phi_f, alpha

########################################################################################################################
# Main Script
########################################################################################################################

def main():

    # Step 1: Create an input array with the reactant gas state
    initialize_parameters(
        T=503.15,
        P=10.0 * 100000,
        Phi=1.0,
        Fuel='H2',
        mech='Li-Dryer-H2-mechanism.yaml',
    )

    # Step 1: Collect all the present PeleC data directories
    dir_path = os.path.dirname(os.path.realpath(__file__))

    time_data_dir = [os.path.join(dir_path, raw_data_folder, time_step)
                     for raw_data_folder in os.listdir(dir_path)
                     if os.path.isdir(os.path.join(dir_path, raw_data_folder)) and raw_data_folder.startswith(f'Raw-{data_set}')
                     for time_step in os.listdir(os.path.join(dir_path, raw_data_folder))
                     if os.path.isdir(os.path.join(dir_path, raw_data_folder, time_step)) and time_step.startswith('plt')]

    # Step 2: Chronologically order the pltFiles and truncate the raw data list if skip loading is enabled
    updated_data_list = sort_files(time_data_dir)
    updated_data_list = [updated_data_list[0]]

    # Step 3:
    ddt_box_size = 1e-2  # cm
    ddt_y_loc = 0.0462731
    # If multiple plot files are provided, determine the size of the box to fit data from the first and last file
    if np.array(updated_data_list).ndim > 1:
        initial_box = flame_contour(updated_data_list[0], y_loc=ddt_y_loc, box_bnds=True, box_size=ddt_box_size)
        final_box = flame_contour(updated_data_list[-1], y_loc=ddt_y_loc, box_bnds=True, box_size=ddt_box_size)
    else:
        initial_box = final_box = flame_contour(updated_data_list[0], y_loc=ddt_y_loc, box_bnds=True, box_size=ddt_box_size)

    x_bnds, y_bnds = [np.minimum(initial_box[0], final_box[0]), np.maximum(initial_box[1], final_box[1])]

    # Step 4: Extract the data from the plot files in parallel
    print('Beginning Data Import')
    collective_data = parallel_processing_function(updated_data_list, (x_bnds, y_bnds,), pelec_flame_data)
    print('Completed Data Import')

    # Step 5: Extract the flame normal and calculate the CEMA jacobian along the progress variable ordinate
    for data in collective_data:
        # Calculate the CEMA source terms
        print('Calculating CEMA Source Terms')
        g_w, g_d, g_r, g_T = cema_source_terms(data)

        # Determine the CEMA jacobian
        print('Calculating CEMA Jacobian')
        flame_pts, jacobian_arr = cema_jacobian(data, g_w)

        # Calculate the
        phi_w, phi_f, alpha = cema_mode(data, jacobian_arr, g_w, g_d, g_r, g_T, flame_pts)

        # Plot the
        x_arr = np.unique(data['x'])[~np.isnan(np.unique(data['x']))]
        y_arr = np.unique(data['y'])[~np.isnan(np.unique(data['y']))]

        plt.figure(figsize=(8, 6))
        cmap = mcolors.LinearSegmentedColormap.from_list("custom_rgb", ["red", "red", "blue", "green", "green"], N=256)
        norm = mcolors.LogNorm(vmin=0.1, vmax=1000)
        # Use pcolormesh for uniform grids
        mesh = plt.pcolormesh(x_arr, y_arr, data['Temp'], shading='auto', cmap='gray')
        # Scatter plot for the normal line
        scatter = plt.scatter(flame_pts[:, 0], flame_pts[:, 1], c=phi_w)

        cbar = plt.colorbar(mesh)
        cbar.set_label("Temperature [K]")
        cbar = plt.colorbar(scatter)
        cbar.set_label("Phi_w")
        plt.title("Phi_w")
        plt.show()

        plt.figure(figsize=(8, 6))
        cmap = mcolors.LinearSegmentedColormap.from_list("custom_rgb", ["red", "red", "blue", "green", "green"], N=256)
        norm = mcolors.LogNorm(vmin=0.1, vmax=1000)
        # Use pcolormesh for uniform grids
        mesh = plt.pcolormesh(x_arr, y_arr, data['Temp'], shading='auto', cmap='gray')
        # Scatter plot for the normal line
        scatter = plt.scatter(flame_pts[:, 0], flame_pts[:, 1], c=phi_f)

        cbar = plt.colorbar(mesh)
        cbar.set_label("Temperature [K]")
        cbar = plt.colorbar(scatter)
        cbar.set_label("Phi_f")
        plt.title("Phi_f")
        plt.show()

        plt.figure(figsize=(8, 6))
        # Use pcolormesh for uniform grids
        mesh = plt.pcolormesh(x_arr, y_arr, data['Temp'], shading='auto', cmap='gray')
        # Scatter plot for the normal line
        scatter = plt.scatter(flame_pts[:, 0], flame_pts[:, 1], c=alpha, cmap=cmap, norm=norm)

        cbar = plt.colorbar(mesh)
        cbar.set_label("Temperature [K]")
        cbar = plt.colorbar(scatter)
        cbar.set_label("Phi_f")
        plt.title("alpha")
        plt.show()

    return

if __name__ == "__main__":
    main()