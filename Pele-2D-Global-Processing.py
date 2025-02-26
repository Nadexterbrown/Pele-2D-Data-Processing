import os, yt, multiprocessing, re, time, textwrap, itertools, traceback, sys
from scipy.interpolate import RegularGridInterpolator, griddata
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import cKDTree
from scipy.stats import linregress
from scipy.signal import savgol_filter
from sdtoolbox.thermo import soundspeed_fr
from matplotlib.ticker import ScalarFormatter
import matplotlib.animation as animation
from matplotlib.tri import Triangulation
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cantera as ct
import numpy as np
import pandas as pd

yt.set_log_level(0)

########################################################################################################################
# Global Program Setting Variables
########################################################################################################################
version = 34

flame_thickness_bin_size = 11
flame_temp = 2500
plotting_bnds_bin = 0
n_procs = 24

data_set = 'PeleC'

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
        results = []
        for i in range(len(iter_arr)):
            results.append(worker_function((iter_arr[i], const_list, input_params, predicate, kwargs)))

    return results

def init_pool(global_params):
    """Initializer function to set the global variable."""
    global input_params
    input_params = global_params

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
# Function Scripts - Data Smoothing
########################################################################################################################

def data_smoothing_function(data_version, bin_size=51):
    ###########################################
    # Internal Functions
    ###########################################
    def load_smoothing_data(file_path):
        """
        Load data from the specified file and extract headers from the second commented line.

        Parameters:
            file_path (str): Path to the data file.

        Returns:
            pd.DataFrame: Loaded data with properly extracted headers.
        """
        # Read the file, using the first line as comments and the second line as headers
        with open(file_path, 'r') as file:
            # Read the second commented line (header line)
            header_line = None
            comment_lines = 0
            for line in file:
                if line.startswith('#'):
                    comment_lines += 1
                    if comment_lines == 2:  # Second commented line
                        header_line = line.strip('#').strip()  # Remove '#' and extra spaces
                        break

        # Column width
        column_width = 55

        # Split the header line into individual headers based on the column width
        headers = [header_line[i:i + column_width].strip() for i in range(0, len(header_line), column_width)]

        # Load the data, skipping the comment lines and using the extracted headers
        data = pd.read_csv(file_path, sep=r'\s+', comment='#', header=None, names=headers)

        return data

    def data_smoothing_function(data, bin_size=51):
        """
        Smooth all columns (except the time column) using a first-order polynomial fit over a bin size.
        The smoothing will start after applying the half-bin offset to the time data.

        Args:
            data (pd.DataFrame): DataFrame where the first column is time, and the rest are data columns.
            bin_size (int): The size of the bin for polynomial fitting (must be odd).

        Returns:
            pd.DataFrame: DataFrame with smoothed data for all columns.
        """
        if bin_size % 2 == 0:
            raise ValueError("Bin size must be odd to center around a cell.")

        # Ensure 'Time [s]' is the first column
        time_column = 'Time [s]'  # Adjust this if the time column has a different name
        smoothed_data = pd.DataFrame(columns=data.columns)

        half_bin = bin_size // 2
        columns_to_smooth = [col for col in data.columns if col != time_column]

        save_time = True
        time_array = []
        for value_column in columns_to_smooth:
            smoothed_values = []
            for i in range(half_bin, len(data) - half_bin):  # Start from half_bin and end at len(data) - half_bin
                # Define the bin range
                start_idx = i - half_bin
                end_idx = i + half_bin + 1

                # Select data within the bin
                bin_data = data.iloc[start_idx:end_idx]

                temp_time = bin_data[time_column].values
                temp_data = bin_data[value_column].values

                # Remove NaN values from temp_data (value_column) only
                valid_idx = ~np.isnan(temp_data)
                temp_time = temp_time[valid_idx]
                temp_data = temp_data[valid_idx]

                # Perform the first-order polynomial fit if there are enough valid points
                if len(temp_data) >= 2:
                    x = temp_time
                    y = temp_data
                    coeffs = np.polyfit(x, y, 1)
                    poly = np.poly1d(coeffs)

                    # Get the smoothed value at the current time value
                    smoothed_values.append(poly(data[time_column].iloc[i]))
                else:
                    # If not enough valid points, append the original value
                    smoothed_values.append(data[value_column].iloc[i])

                if save_time:
                    time_array.append(data[time_column].iloc[i])

            save_time = False

            # Replace the original column with smoothed values
            smoothed_data[value_column] = smoothed_values

        # Add the time column to the smoothed data
        smoothed_data[time_column] = time_array

        return smoothed_data

    def save_smoothed_data(smoothed_data, output_file, input_file=None):
        """
        Save the smoothed data to a file while preserving column orientation,
        ensuring columns are 55 characters wide, and using the same header as the input file.

        Parameters:
            smoothed_data (pd.DataFrame): Smoothed data.
            output_file (str): Path to save the output file.
            input_file (str, optional): Path to the input file (used to get the original headers).
        """
        # Load headers from the input file if provided
        if input_file:
            with open(input_file, 'r') as file:
                # Skip non-header lines to get the second header line (commented line)
                comment_lines = 0
                header_line = None
                for line in file:
                    if line.startswith('#'):
                        comment_lines += 1
                        if comment_lines == 2:  # Second commented line
                            header_line = line.strip('#').strip()  # Remove '#' and extra spaces
                            break

            # Column width
            column_width = 55

            # Split the header line into individual headers based on the column width
            headers = [header_line[i:i + column_width].strip() for i in range(0, len(header_line), column_width)]

            # Update the column headers of the smoothed data with the extracted headers
            smoothed_data.columns = headers

        # Ensure that the data is written with exactly 55-character wide columns
        with open(output_file, 'w') as f:
            # Write the header
            f.write(' '.join([f"{col:<55}" for col in smoothed_data.columns]) + '\n')

            # Write the data
            for _, row in smoothed_data.iterrows():
                f.write(' '.join([f"{val:<55}" for val in row]) + '\n')

    ###########################################
    # Main Function
    ###########################################
    # Step 0: Define the vertical location if more than one folder exists
    vertical_location = None
    # Step 1: Define the input and output file paths
    dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            f"Processed-Global-Results-V{data_version}")
    # List subdirectories in the given path
    subdirectories = [d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]

    if len(subdirectories) == 1 and vertical_location is None:
        # Use the only folder if no domain_info is provided
        domain_info_folder = subdirectories[0]
        vertical_location = float(domain_info_folder.split('-')[-1].replace('cm', '').replace('y', ''))
    elif vertical_location is None:
        raise ValueError("Multiple folders exist in the directory. 'domain_info' must be provided.")

    # Construct the output directory path
    data_dir_path = os.path.join(
        dir_path,
        f"y-{vertical_location:.3g}cm"
    )

    # Step 2:
    raw_data = load_smoothing_data(
        ensure_long_path_prefix(os.path.join(data_dir_path, f"Wave-Tracking-Results-V{data_version}.txt")))

    # Step 3: Smooth the data
    smoothed_data = data_smoothing_function(raw_data)

    # Step 4: Save the smoothed data
    output_file = ensure_long_path_prefix(
        os.path.join(data_dir_path, f"Wave-Tracking-Smoothed-Results-V{data_version}.txt"))
    save_smoothed_data(smoothed_data, output_file)

    return

########################################################################################################################
# Function Scripts - Data Extraction
########################################################################################################################

def domain_size_parameters(directory_path, desired_y_location):
    """
    Extract domain size parameters from the given directory path and desired y-location.

    :param directory_path: Path to the directory containing the data.
    :param desired_y_location: Desired y-location (can be a string or a float).
    :return: Tuple containing domain size parameters.
    """
    # Load the data for physical size extraction
    ds = yt.load(directory_path)
    max_level = ds.index.max_level

    # Step 2:
    data = ds.covering_grid(level=max_level,
                            left_edge=[0.0, 0.0, 0.0],
                            dims=ds.domain_dimensions * [2 ** max_level, 2 ** max_level, 1],
                            # And any fields to preload (this is optional!)
                            # fields=desired_varables
                            )



    # Create a covering grid at the maximum level present
    x_coords = np.arange(ds.domain_left_edge[0].to_value(), ds.domain_right_edge[0].to_value(), data.dds[0].to_value())
    y_coords = np.arange(ds.domain_left_edge[1].to_value(), ds.domain_right_edge[1].to_value(), data.dds[1].to_value())
    grid_arr = np.array([x_coords, y_coords], dtype=object)

    # Access the grid for the highest level
    highest_level_grids = [grid for grid in ds.index.grids if grid.Level == max_level]
    # Extract y values from these grids
    y_values = []
    for grid in highest_level_grids:
        y_values.extend(grid['boxlib', 'y'].to_value().flatten())
    # Get unique y values and store them in an array
    y_arr = np.unique(y_values)

    if isinstance(desired_y_location, str):
        if desired_y_location == "Bottom":
            y_slice_index = 0
            y_slice_loc = data.LeftEdge[1].to_value()
        elif desired_y_location == "Top":
            y_slice_index = data.ActiveDimensions[1] - 1
            y_slice_loc = data.RightEdge[1].to_value()
        elif desired_y_location == "DDT":
            y_slice_index = np.unravel_index(np.argmax(data['boxlib', 'Temp'].to_value(), axis=None),
                                             data['boxlib', 'Temp'].to_value().shape)[1]
            y_slice_loc = data['boxlib', 'y'][0][y_slice_index].to_value()[0]
        else:
            y_slice_index = data.ActiveDimensions[1] // 2 - 1
            y_slice_loc = y_arr[y_slice_index]
    else:
        y_slice_index = np.argmin(abs(y_arr - desired_y_location))
        y_slice_loc = y_arr[y_slice_index]

    return (np.array([[0, y_slice_index], [data.ActiveDimensions[0], y_slice_index]]),
            np.array([[data.LeftEdge[0].to_value(), y_slice_loc], [data.RightEdge[0].to_value(), y_slice_loc]]),
            grid_arr)

def local_data_refinement(raw_data, domain_info, required_vars, preloaded_grids=None):
    """
    Extracts refined data for multiple required variables.

    Parameters:
        raw_data: The full simulation dataset.
        domain_info: Information about the domain (e.g., y-level for extraction).
        required_vars: A list of variable names that need to be loaded.
        preloaded_grids: Optional dictionary of preloaded grids for each refinement level.

    Returns:
        A dictionary of {mapped_variable_name: (x_sorted, var_sorted)}
    """
    VAR_NAME_MAP = {
        'Temp': 'Temperature',
        'pressure': 'Pressure',
        'density': 'Density',
        'soundspeed': 'Sound Speed',
        'x_velocity': 'Velocity',
        'heatRelease': 'Heat Release Rate',
        'viscosity': 'Viscosity',
    }

    # Get the maximum refinement level and smallest grid spacing
    max_level = raw_data.index.max_level
    dx_min = raw_data.index.get_smallest_dx()
    # Dictionary to store refined data per level
    level_data = {level: {var: {} for var in required_vars} for level in range(max_level + 1)}
    used_grids = {} if preloaded_grids is None else preloaded_grids
    # If grids are preloaded, process only those levels
    levels_to_process = reversed(preloaded_grids.keys()) if preloaded_grids else range(max_level, -1, -1)
    for level in levels_to_process:
        dx = dx_min.to_value() * 2 ** (max_level - level)  # Grid spacing at this level
        # If preloaded_grids is provided, only use those specific grids
        if preloaded_grids:
            grids = [
                grid for grid in raw_data.index.grids
                if any(grid.id == g.id for g in preloaded_grids[level])
            ]
        else:
            grids = [grid for grid in raw_data.index.grids if grid.Level == level]
        for grid in grids:
            x = grid["boxlib", "x"].to_value().flatten()
            y = grid["boxlib", "y"].to_value().flatten()
            # Find points that match the target y-level
            mask = np.isclose(y, domain_info[1][0][1], atol=dx)  # Allow small tolerance
            if np.any(mask) and not preloaded_grids:
                used_grids.setdefault(level, []).append(grid)  # Store grids only if not preloaded
            # Process each required variable
            for var in required_vars:
                try:
                    temp_var = grid["boxlib", var].flatten()
                    for xi, vi in zip(x[mask], temp_var[mask]):
                        level_data[level][var][xi] = vi  # Higher levels overwrite lower ones
                except:
                    continue
    # Convert to sorted NumPy arrays and apply variable name mapping for each level
    final_data = {
        level: {
            VAR_NAME_MAP.get(var, var): (
                np.array(sorted(level_data[level][var].keys())),
                np.array([level_data[level][var][xi] for xi in sorted(level_data[level][var].keys())])
            )
            for var in required_vars
        }
        for level in range(max_level + 1)
    }

    # Merge the data from all levels into a single dataset
    merged_data = {}
    # Iterate over each variable in final_data
    for var in final_data[0].keys():
        # Initialize merged x and y arrays with the data from level 0
        merged_x = np.array(sorted(final_data[0][var][0]))
        merged_y = np.array([final_data[0][var][1][np.argmin(np.abs(final_data[0][var][0] - x))] for x in merged_x])
        # Process levels starting from level 1 upwards
        for level in range(1, len(final_data)):
            dx = dx_min.to_value() * 2 ** (max_level - level)  # Grid spacing at this level

            x_next_level, y_next_level = final_data[level][var]
            # Skip if x_next_level or y_next_level are empty
            if x_next_level.size == 0 or y_next_level.size == 0:
                # print(f"Warning: Skipping level {level} for variable {var} due to empty data.")
                continue
            # Break up the next level into chunks based on the grid spacing
            split_indices = np.where(np.diff(x_next_level) > 2 * dx)[0] + 1
            x_chunks = np.array_split(x_next_level, split_indices)
            y_chunks = np.array_split(y_next_level, split_indices)
            # Iterate over x_next_level to insert each value into merged_x
            for i in range(len(x_chunks)):
                x_min = x_chunks[i][0]
                x_max = x_chunks[i][-1]
                # Check where x_min fits in merged_x and find the nearest value
                insert_index_min = np.searchsorted(merged_x, x_min)
                # Check where x_max fits in merged_x and find the nearest value
                insert_index_max = np.searchsorted(merged_x, x_max)
                # Remove any overlapping section in merged_x around x_min and x_max
                merged_x = np.delete(merged_x, np.s_[insert_index_min:insert_index_max])
                merged_y = np.delete(merged_y, np.s_[insert_index_min:insert_index_max])
                # Insert the full array of x_chunks[i] and corresponding y_chunks[i] values
                merged_x = np.insert(merged_x, insert_index_min, x_chunks[i])
                merged_y = np.insert(merged_y, insert_index_min, y_chunks[i])

        # Store the merged data for the variable
        merged_data[var] = (merged_x, merged_y)

        plt_check = False
        if plt_check:
            plt.figure(figsize=(8, 6))
            #plt.plot(final_data[6]['Temperature'][0], final_data[6]['Temperature'][1], 'k-', label='Level 6')
            #plt.plot(final_data[5]['Temperature'][0], final_data[5]['Temperature'][1], 'b.', label='Level 5')
            #plt.plot(final_data[4]['Temperature'][0], final_data[4]['Temperature'][1], 'g.', label='Level 4')
            plt.plot(final_data[3]['Temperature'][0], final_data[3]['Temperature'][1], 'y.', label='Level 3')
            plt.plot(final_data[3]['Temperature'][0], final_data[2]['Temperature'][1], 'y.', label='Level 2')
            plt.plot(final_data[3]['Temperature'][0], final_data[1]['Temperature'][1], 'y.', label='Level 1')
            plt.plot(final_data[3]['Temperature'][0], final_data[0]['Temperature'][1], 'y.', label='Level 0')
            plt.plot(merged_data['Temperature'][0], merged_data['Temperature'][1], 'r--', label='Merged')
            plt.xlabel('X Values')
            plt.xlim(140, 220)
            plt.legend()
            plt.grid(True)
            plt.show()

    return merged_data, used_grids

def load_plt_data(file_path, domain_info, CHECK_FLAGS):
    # Step 1: Load the data using yt
    raw_data = yt.load(file_path)

    # Step 2:
    CATEGORY_LOAD_MAP = {
        'Flame': 'Temp',
        'Leading Shock': 'pressure',
        'Maximum Pressure': 'pressure',
        'Pre-Shock': 'pressure',
        'Post-Shock': 'pressure',
    }

    SUB_CATEGORY_MAP = {
        'Temperature': 'Temp',
        'Pressure': 'pressure',
        'Velocity': 'x_velocity',
        'Relative Velocity': 'x_velocity',
        'Thermodynamic State': ('Temp', 'pressure', 'density', 'soundspeed'),
        'Heat Release Rate PeleC': 'heatRelease',
        'Reynolds Number': 'viscosity',
    }

    # Dictionary to store values that need to be loaded
    values_to_load = set()

    # Step 1: Check each category
    for category, sub_dict in CHECK_FLAGS.items():
        if isinstance(sub_dict, dict):  # Ensure it's a dictionary
            has_true_value = any(
                (isinstance(value, dict) and value.get('Flag', False)) or value is True
                for value in sub_dict.values()
            )

            if has_true_value:
                # Check if category has a mapped variable
                if category in CATEGORY_LOAD_MAP:
                    values_to_load.add(CATEGORY_LOAD_MAP[category])

                # Step 2: Check subcategories
                for sub_key in sub_dict.keys():
                    if sub_key in SUB_CATEGORY_MAP:
                        sub_value = SUB_CATEGORY_MAP[sub_key]
                        if isinstance(sub_value, tuple):  # Handle multiple variables
                            for sub_sub_value in sub_value:
                                if ("boxlib", sub_sub_value) in raw_data.field_list:
                                    values_to_load.add(sub_sub_value)
                        else:
                            if ("boxlib", sub_value) in raw_data.field_list:
                                values_to_load.add(sub_value)

    # Step 3: Extract the required variables
    data, grids = local_data_refinement(raw_data, domain_info, values_to_load)

    return data, grids


def preload_value_check(var_str, preloaded_values, raw_data, domain_info, grid_arr, position_flag=False):
    ###########################################
    # Internal Functions
    ###########################################
    def get_var_name(value):
        for key, val in VAR_NAME_MAP.items():
            if val == value:
                return key  # Return the corresponding key
        return value  # Return None if value is not found

    ###########################################
    # Main Function
    ###########################################
    """
    Checks if a variable or list of variables is present in preloaded_values.
    If found, returns the corresponding value(s); otherwise, calls local_data_refinement().

    :param var_str: A single string or a list/tuple of strings to check.
    :param preloaded_values: The dictionary or list structure containing preloaded values.
    :param not_found_value: The value to return if a variable is not found (default: None).
    :param raw_data: Data needed for local_data_refinement() if a value is missing.
    :param domain_info: Domain info for local_data_refinement().
    :param load_arr: Load array for local_data_refinement().
    :param grid_arr: Preloaded grids for local_data_refinement().
    :return: A single value if var_str is a string, or a dictionary mapping each variable to its value/not_found_value.
    """
    # Step 1:
    VAR_NAME_MAP = {
        'Temperature': 'Temp',
        'Pressure': 'pressure',
        'Density': 'density',
        'Sound Speed': 'soundspeed',
        'Velocity': 'x_velocity',
        'Heat Release Rate': 'heatRelease',
        'Viscosity': 'viscosity',
    }

    # Step 2:
    results = {}
    missing_var = []
    if isinstance(var_str, (tuple, list)):
        # Step 3: Check each variable in the list
        for var in var_str:
            if preloaded_values is not None and var in preloaded_values[0]:
                results[var] = preloaded_values[preloaded_values[0].index(var) + 1]

            elif var in VAR_NAME_MAP.keys() and ("boxlib", VAR_NAME_MAP.get(var)) in raw_data.field_list:
                missing_var.append(VAR_NAME_MAP.get(var))

            elif ("boxlib", var) in raw_data.field_list:
                missing_var.append(var)

            else:
                results[var] = None

        # Step 4:
        if missing_var:
            if 'Position' in missing_var:
                missing_var.remove('Position')
                position_flag = True

            tmp_arr, _ = local_data_refinement(raw_data, domain_info, missing_var, preloaded_grids=grid_arr)
            for var in missing_var:
                if var in VAR_NAME_MAP.values():
                    results[get_var_name(var)] = tmp_arr[get_var_name(var)][1]
                else:
                    results[var] = tmp_arr[var][1]

            if position_flag:
                if missing_var:
                    return tmp_arr[var_str[0]][0], results
                else:
                    return preloaded_values[1], results
            else:
                return results

        else:
            if position_flag:
                if preloaded_values is not None:
                    return preloaded_values[1], results
                else:
                    tmp_arr, _ = local_data_refinement(raw_data, domain_info, missing_var, preloaded_grids=grid_arr)
                    return tmp_arr[var_str[0]][0], results
            else:
                return results

    # Step 2: Check if the variable is in preloaded_values
    elif isinstance(var_str, str):
        if preloaded_values is not None and var_str in preloaded_values[0]:
            results[var_str] = preloaded_values[preloaded_values[0].index(var_str) + 1]
            return results

        elif var_str in VAR_NAME_MAP.keys() and ("boxlib", VAR_NAME_MAP.get(var_str)) in raw_data.field_list:
            missing_var = VAR_NAME_MAP.get(var_str)

        elif ("boxlib", var_str) in raw_data.field_list:
            missing_var = var_str

        else:
            results[var_str] = None

        # Step 4:
        if missing_var:
            if 'Position' in missing_var:
                missing_var.remove('Position')
                position_flag = True

            tmp_arr, _ = local_data_refinement(raw_data, domain_info, missing_var, preloaded_grids=grid_arr)
            if var_str in VAR_NAME_MAP.values():
                results[get_var_name(var_str)] = tmp_arr[var_str][1]
            else:
                results[var_str] = tmp_arr[var_str][1]

            if position_flag:
                if missing_var:
                    return tmp_arr[var_str][0], results
                else:
                    return preloaded_values[1], results
            else:
                return results

    else:
        raise TypeError("var_str must be a string, list, or tuple")

def convert_units(value, type: str):
    if type == 'Position' or type == 'x' or type == 'y' or type == 'Flame Thickness' or type == 'Surface Length':
        return value / 100
    elif type == 'Temperature' or type == 'Temp':
        return value
    elif type == 'Pressure' or type == 'pressure':
        return value / 10
    elif type == 'Density' or type == 'density':
        return value * 1000
    elif type == 'Velocity' or type == 'Relative Velocity' or type == 'x_velocity' or type == 'y_velocity':
        return value / 100
    elif type == 'Heat Release Rate Cantera':
        return value
    elif type == 'Heat Release Rate PeleC':
        return value
    elif value is None:
        return np.nan
    else:
        return value

########################################################################################################################
# Function Scripts - Data Plotting
########################################################################################################################

def plt_var_bnds(args):
    ###########################################
    # Main Function
    ###########################################
    global input_params
    iter_var, const_arr, kwargs = args

    temp_plt_files = iter_var
    domain_info = kwargs.get('domain_info', [])
    CHECK_FLAGS = kwargs.get('CHECK_FLAGS', {})

    # Step 1: Extract the keys where the value is True (including support for combined flags)
    keys_with_true_values = [
        key for key, value in CHECK_FLAGS.get('Domain State Animations', {}).items()
        if key not in {'Combined', 'Flame Thickness', 'Surface Contour'}  # Exclude specific keys
           and isinstance(value, dict)  # Ensure the value is a dictionary
           and value.get('Flag', False)  # Check if the 'Flag' is True
    ]

    # Initialize the temp_bounds_arr to store bounds for each key
    temp_bounds_arr = np.empty((len(keys_with_true_values), 3), dtype=object)

    # Step 2: Load data from the plot files using yt
    raw_data = yt.load(temp_plt_files)
    processed_data, grids = load_plt_data(temp_plt_files, domain_info, CHECK_FLAGS)

    # Step 3: Loop through each variable and determine bounds
    for i, key in enumerate(keys_with_true_values):
        try:
            if key == 'Temperature':
                temp_arr = processed_data['Temperature'][1]
                value_arr = convert_units(temp_arr, 'Temperature')
            elif key == 'Pressure':
                temp_arr = processed_data['Pressure'][1]
                value_arr = convert_units(temp_arr, 'Pressure')
            elif key == 'Velocity':
                temp_arr = processed_data['Velocity'][1]
                value_arr = convert_units(temp_arr, 'x_velocity')
            elif key == 'Species':
                print('Max Value Determination for Species is W.I.P.')
                continue
            elif key == 'Heat Release Rate Cantera':
                temp_arr, _ = heat_release_rate_extractor('Cantera', raw_data=raw_data, grid_arr=grids, domain_info=domain_info)
                value_arr = convert_units(temp_arr, 'Heat Release Rate Cantera')
            elif key == 'Heat Release Rate PeleC':
                try:
                    temp_arr = processed_data['Heat Release Rate'][1]
                    value_arr = convert_units(temp_arr, 'Heat Release Rate PeleC')
                except:
                    temp_arr, _ = heat_release_rate_extractor('Cantera', raw_data=raw_data, grid_arr=grids, domain_info=domain_info)
                    value_arr = convert_units(temp_arr, 'Heat Release Rate Cantera')
            elif key == 'Reynolds Number':
                temp_arr = reynolds_number_extractor('Flame', raw_data=raw_data, grid_arr=grids, domain_info=domain_info)
                value_arr = convert_units(temp_arr, 'Reynolds Number')
            else:
                continue  # Skip if key is not recognized

            # Step 4: Assign the min and max values for this variable
            temp_bounds_arr[i, 0] = key
            temp_bounds_arr[i, 1] = np.min(value_arr)
            temp_bounds_arr[i, 2] = np.max(value_arr)

        except KeyError as e:
            print(f"Warning: Missing data for {key}. Skipping.")
            continue
        except Exception as e:
            print(f"Error processing {key}: {e}")
            continue

    return temp_bounds_arr

def animation_axis(plt_dirs, ddt_dir, ddt_plt_file, domain_info, CHECK_FLAGS):
    ###########################################
    # Main Function
    ###########################################
    print('Begin Individual Variable Bounds')

    try:
        ddt_idx = plt_dirs.index(os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), ddt_dir, f'Raw-{data_set}-Data', ddt_plt_file)))
    except:
        print('DDT Folder not in current directory, finding appropriate files.')
        temp_dir_path = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), ddt_dir))
        plt_dirs = [os.path.join(temp_dir_path, raw_data_folder, time_step)
                     for raw_data_folder in os.listdir(temp_dir_path)
                     if os.path.isdir(os.path.join(temp_dir_path, raw_data_folder)) and raw_data_folder.startswith(f'Raw-{data_set}')
                     for time_step in os.listdir(os.path.join(temp_dir_path, raw_data_folder))
                     if os.path.isdir(os.path.join(temp_dir_path, raw_data_folder, time_step)) and time_step.startswith('plt')]
        ddt_idx = plt_dirs.index(os.path.abspath(
            os.path.join(os.path.dirname(os.path.realpath(__file__)), ddt_dir, f'Raw-{data_set}-Data', ddt_plt_file)))

    temp_plt_files = plt_dirs[max(0, ddt_idx - plotting_bnds_bin):min(len(plt_dirs), ddt_idx + plotting_bnds_bin + 1)]

    temp_max_arr = parallel_processing_function(temp_plt_files, (),
                                                plt_var_bnds,
                                                domain_info=domain_info,
                                                CHECK_FLAGS=CHECK_FLAGS)

    n_rows = len(temp_max_arr[0])
    final_results = []

    for i in range(n_rows):
        row_values_2nd_col = [temp_max_arr[j][i][1] for j in range(len(temp_max_arr))]
        row_values_3rd_col = [temp_max_arr[j][i][2] for j in range(len(temp_max_arr))]
        row_texts = [temp_max_arr[j][i][0] for j in range(len(temp_max_arr))]

        min_index_2nd_col = row_values_2nd_col.index(min(row_values_2nd_col))
        max_index_3rd_col = row_values_3rd_col.index(max(row_values_3rd_col))

        final_results.append([row_texts[min_index_2nd_col], row_values_2nd_col[min_index_2nd_col], row_values_3rd_col[max_index_3rd_col]])

    print('Completed Individual Variable Bounds')
    return np.array(final_results, dtype=object)

def state_animation(method, **kwargs):
    ###########################################
    # Internal Functions
    ###########################################
    def plot_axis(ax, x_data, y_data, label, linestyle, color, ylabel, ylim=None):
        """Helper function to plot data on a single axis."""
        ax.plot(x_data, y_data, label=label, linestyle=linestyle, color=color)
        ax.set_ylabel(ylabel)
        if ylim is not None:
            ax.set_ylim(*ylim)

    def get_filtered_data(reference_loc, x_data_arr, y_data_arr):
        if reference_loc is not None:
            indices = np.where(
                (x_data_arr >= x_data_arr[reference_loc] - reference_loc / 2) &
                (x_data_arr <= x_data_arr[reference_loc] + reference_loc / 2)
            )[0]
            return x_data_arr[indices], y_data_arr[indices]
        return x_data_arr, y_data_arr

    def plot_single_dataset_frame(reference_loc=None):
        # Step 1: Create the figure and axes
        fig, ax1 = plt.subplots(figsize=(10, 6))
        axes = [ax1]

        # Step 2: Plot the data
        tmp_x_data, tmp_y_data = get_filtered_data(reference_loc, x_data_arr, y_data_arr)

        # Check for log scale (e.g., pressure, heat release rate)
        is_log_scale = var_name in ['Pressure']

        if y_bounds is not None:
            plot_axis(ax=ax1, x_data=tmp_x_data, y_data=tmp_y_data,
                      label=var_name, linestyle='-', color='k',
                      ylabel=var_name, ylim=(y_bounds[0], y_bounds[1]))
        else:
            plot_axis(ax=ax1, x_data=tmp_x_data, y_data=tmp_y_data,
                      label=var_name, linestyle='-', color='k',
                      ylabel=var_name)

        ax1.set_xlabel('Position [cm]')
        ax1.set_ylabel(var_name)
        ax1.grid(True, axis='x')

        # Apply log scale if necessary
        ax1.set_yscale('log' if is_log_scale else 'linear')

        # Step 3: Set the title and axis labels
        title = f"{var_name} Variation at y = {domain_size[1][1][1]} cm and t = {time} s"
        wrapped_title = "\n".join(textwrap.wrap(title, width=55))
        plt.suptitle(wrapped_title, ha="center")

        ax1.set_xlim(x_bounds if x_bounds is not None else (0, domain_size[1][1][0]))

        # Step 4: Create filename and save plot
        formatted_time = f"{time:.16f}".rstrip('0').rstrip('.')
        filename = os.path.join(output_dir_path, f"{var_name}-Animation-{plt_folder}.png")

        plt.tight_layout()
        plt.savefig(filename, format='png')
        plt.close()

    def plot_collective_dataset_frame(reference_loc=None):
        # Step 1: Create the figure and axes
        fig, ax1 = plt.subplots(figsize=(12, 8))  # Increased figure size for better spacing
        axes = [ax1]

        # Step 2: Plot each variable on its respective axis
        for i, obj in enumerate(var_name):
            tmp_x_data, tmp_y_data = get_filtered_data(reference_loc, x_data_arr, y_data_arr)

            # Determine if log scale is needed
            is_log_scale = obj in ['Pressure']

            if i == 0:
                # Plot the first variable on the main axis
                if y_bounds is not None:
                    plot_axis(ax=axes[0], x_data=tmp_x_data, y_data=tmp_y_data[i], label=obj, linestyle='-', color='k',
                              ylabel=obj, ylim=(y_bounds[i][0], y_bounds[i][1]))
                else:
                    plot_axis(ax=axes[0], x_data=tmp_x_data, y_data=tmp_y_data[i], label=obj, linestyle='-', color='k',
                              ylabel=obj)
                ax1.set_xlabel('Position [cm]')
                ax1.set_ylabel(obj)
                ax1.grid(True, axis='x')
                ax1.set_yscale('log' if is_log_scale else 'linear')  # Apply log scale if necessary
            else:
                # Create a new y-axis for additional variables
                ax = ax1.twinx()
                ax.spines["right"].set_position(("outward", 60 * (i - 1)))  # Offset each new axis outward
                if y_bounds is not None:
                    plot_axis(ax=ax, x_data=tmp_x_data, y_data=tmp_y_data[i], label=obj, linestyle='--', color=f"C{i}",
                              ylabel=obj, ylim=(y_bounds[i][0], y_bounds[i][1]))
                else:
                    plot_axis(ax=ax, x_data=tmp_x_data, y_data=tmp_y_data[i], label=obj, linestyle='--', color=f"C{i}",
                              ylabel=obj)
                ax.set_yscale('log' if is_log_scale else 'linear')  # Apply log scale if necessary
                axes.append(ax)

        # Step 3: Adjust right-side y-axis label positions to prevent overlap
        for j, ax in enumerate(axes[1:], start=1):  # Iterate over the additional axes
            # Manually set the position of each right y-axis
            manual_position = 1.01 + 0.2 * (j - 1)  # Adjust this formula or use fixed values for specific positions
            ax.spines["right"].set_position(("axes", manual_position))  # Set axis position relative to plot area

        # Step 4: Add the title
        title = " and ".join(var_name) + f" Variation at y = {domain_size[1][1][1]} cm and t = {time} s"
        wrapped_title = "\n".join(textwrap.wrap(title, width=55))
        plt.suptitle(wrapped_title, ha="center")

        # Step 5: Set the x-axis limits
        ax1.set_xlim(x_bounds if x_bounds is not None else (0, domain_size[1][1][0]))

        formatter = ScalarFormatter(useOffset=False, useMathText=False)
        formatter.set_scientific(False)
        ax.xaxis.set_major_formatter(formatter)
        ax.ticklabel_format(style='plain', axis='x')

        # Step 6: Add a legend outside the plot area
        handles, labels = ax1.get_legend_handles_labels()  # Collect handles and labels from the first axis
        for ax in axes[1:]:  # For each additional axis, collect handles and labels
            h, l = ax.get_legend_handles_labels()
            handles.extend(h)
            labels.extend(l)
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.01), ncol=len(var_name),
                   title="Variables")

        # Step 7: Adjust layout and save the plot
        plt.tight_layout()  # Leave space for title and legend

        formatted_time = f"{time:.16f}".rstrip('0').rstrip('.')
        # Step 8: Save the figure
        filename = os.path.join(output_dir_path, f"{'-'.join(var_name)}-Animation-{plt_folder}.png")
        plt.savefig(filename, format='png', bbox_inches='tight')
        plt.close()

    def create_animation():
        # Extract numeric parts of filenames for sorting
        def extract_frame_number(filename):
            match = re.search(r'plt(\d+)', filename)
            return int(match.group(1)) if match else -1

        # Filter and sort files by frame number
        image_files = [
            os.path.join(folder_path, f)
            for f in sorted(os.listdir(folder_path), key=extract_frame_number)
            if f.endswith('.png')  # Only include PNG files
        ]

        if not image_files:
            print("No image files found in the folder.")
            return

        # Load the first image to determine figure size
        first_image = mpimg.imread(image_files[0])
        fig = plt.figure(figsize=(first_image.shape[1] / 100, first_image.shape[0] / 100), dpi=100)
        plt.axis('off')
        img_display = plt.imshow(first_image)

        # Animation update function
        def update(frame):
            img_display.set_array(mpimg.imread(image_files[frame]))
            return img_display,

        # Create and save the animation
        ani = animation.FuncAnimation(fig, update, frames=len(image_files), blit=True)
        writer = animation.FFMpegWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)
        ani.save(animation_filename, writer=writer)
        plt.close(fig)

    ###########################################
    # Main Function
    ###########################################
    time = kwargs.get('time', 0)
    x_data_arr = kwargs.get('x_data_arr', [])
    y_data_arr = kwargs.get('y_data_arr', [])
    reference_loc = kwargs.get('reference_loc', None)
    x_bounds = kwargs.get('x_bounds', None)
    y_bounds = kwargs.get('y_bounds', None)
    domain_size = kwargs.get('domain_size', [])
    var_name = kwargs.get('var_name', "")
    plt_folder = kwargs.get('plt_folder', "")
    folder_path = kwargs.get('folder_path', "")
    output_dir_path = kwargs.get('output_dir_path', "")
    animation_filename = kwargs.get('animation_filename', "")

    if method == 'Plot':
        if isinstance(var_name, str):
            plot_single_dataset_frame(reference_loc=reference_loc)
        else:
            plot_collective_dataset_frame(reference_loc=reference_loc)
    elif method == 'Animate':
        create_animation()
    else:
        print('Error: Did not define viable method argument (Plot or Animate)')

########################################################################################################################
# Function Scripts - Data Processing
########################################################################################################################

############################################################
# Wave Processing Functions
############################################################

def wave_tracking(wave_type, **kwargs):
    ###########################################
    # Internal Functions
    ###########################################
    def find_wave_index(wave_type, data_str):
        if wave_type == 'Flame':
            tmp_arr = np.zeros(len(data_str), dtype=int)
            try:
                tmp_arr[0] = np.argwhere(data_arr['Temperature'] >= flame_temp)[-1]
                tmp_arr[1] = np.argmax(data_arr['Y(HO2)'])

                if abs(tmp_arr[0] - tmp_arr[1]) > 10:
                    print('Warning: Flame Location differs by more than 10 cells!\n'
                            'Flame Temperature Location', x_arr[tmp_arr[0]],
                            '\nFlame Species Location', x_arr[tmp_arr[1]])
                return tmp_arr[1]
            except:
                print('Error: Could not find Y_H to determine Flame Location!')
                tmp_arr[0] = np.argwhere(data_arr['Temperature'] >= flame_temp)[-1]
                return tmp_arr[0]
        elif wave_type == 'Maximum Pressure':
            return np.argmax(data_arr['Pressure'])
        elif wave_type == 'Leading Shock':
            return np.argwhere(data_arr['Pressure'] >= 1.01 * data_arr['Pressure'][-1])[-1][0]
        else:
            raise ValueError('Invalid Wave Type! Must be Flame, Maximum Pressure, or Leading Shock')

    ###########################################
    # Main Function
    ###########################################
    # Step 1: Parse input arguments
    raw_data = kwargs.get('raw_data')
    grid_arr = kwargs.get('grid_arr')
    domain_info = kwargs.get('domain_info')
    pre_loaded_data = kwargs.get('pre_loaded_data')

    # Step 2: Define wave type and data string relations
    WAVE_TYPE_MAP = {
        'Flame': {
            'PeleC': ('Temp', 'Y(HO2)'),
            'Pre Loaded': ('Temperature', 'Y(HO2)')
        },
        'Maximum Pressure': {
            'PeleC': 'pressure',
            'Pre Loaded': 'Pressure'
        },
        'Leading Shock': {
            'PeleC': 'pressure',
            'Pre Loaded': 'Pressure'
        }
    }

    data_str = WAVE_TYPE_MAP[wave_type]['Pre Loaded'] if pre_loaded_data is not None else WAVE_TYPE_MAP[wave_type]['PeleC']

    # Step 3: Extract the data for the wave type
    if pre_loaded_data is not None:
        x_arr = pre_loaded_data[pre_loaded_data[0].index('Position') + 1]
        data_arr = preload_value_check(data_str, pre_loaded_data, raw_data, domain_info, grid_arr)
    else:
        x_arr, data_arr = preload_value_check(data_str, pre_loaded_data, raw_data, domain_info, grid_arr, position_flag=True)

    # Step 4: Find the index of the wave
    wave_idx = find_wave_index(wave_type, data_str)

    return wave_idx, x_arr[wave_idx]

############################################################
# Thermodynamic Processing Functions
############################################################

def thermodynamic_state_extractor(wave_type, **kwargs):
    ###########################################
    # Internal Functions
    ###########################################
    def probe_array():
        wave_loc_adjustments = {
            'Flame': 1e-3,
            'Burned Gas': -1e-3,
            'Maximum Pressure': 0,
            'Pre-Shock': 1e-3,
            'Post-Shock': -1e-3
        }

        if wave_type not in wave_loc_adjustments:
            raise ValueError('Invalid Wave Type!')

        return np.argwhere(x_arr >= (wave_loc + wave_loc_adjustments[wave_type]))[0][0]

    def cantera_soundspeed(data_arr):
        # Step 1: Set Gas Object and Species Composition
        gas_obj = ct.Solution(input_params.mech)

        species_name_list = [f"Y({species})" for species in input_params.species]
        tmp_arr, _ = local_data_refinement(raw_data, domain_info, species_name_list, preloaded_grids=grid_arr)
        species_comp = {input_params.species[i]: tmp_arr[f"Y({input_params.species[i]})"][1][probe_idx] for i in range(len(input_params.species))}

        species_values = {key: arr for key, arr in species_comp.items()}

        gas_obj.TPY = (data_arr[0],
                       data_arr[1] / 10,
                       species_values
        )

        return soundspeed_fr(gas_obj)

    ###########################################
    # Main Function
    ###########################################
    # Step 1: Parse input arguments
    wave_loc = kwargs.get('wave_loc')
    raw_data = kwargs.get('raw_data')
    grid_arr = kwargs.get('grid_arr')
    domain_info = kwargs.get('domain_info')
    pre_loaded_data = kwargs.get('pre_loaded_data')

    # Step 2: Extract the thermodynamic state
    if pre_loaded_data is not None:
        x_arr = pre_loaded_data[pre_loaded_data[0].index('Position') + 1]
        tmp_arr = preload_value_check(['Temperature', 'Pressure', 'Density', 'Sound Speed'], pre_loaded_data,
                                       raw_data, domain_info, grid_arr)
    else:
        x_arr, tmp_arr = preload_value_check(['Temperature', 'Pressure', 'Density', 'Sound Speed'],
                                              pre_loaded_data, raw_data, domain_info, grid_arr, position_flag=True)

    # Step 3: Determine the location of the wave
    try:
        probe_idx = probe_array()
    except Exception as e:
        print(f"Error Could not find thermodynamic location: {e}")
        probe_idx = -1

    # Step 4: Return the thermodynamic state
    data_arr = np.empty(4, dtype=object)
    data_arr[0] = tmp_arr['Temperature'][probe_idx]
    data_arr[1] = tmp_arr['Pressure'][probe_idx]
    data_arr[2] = tmp_arr['Density'][probe_idx]

    # Step 5: Check if sound speed is missing
    if data_arr[3] is None:
        data_arr[3] = cantera_soundspeed(data_arr)
    else:
        data_arr[3] = tmp_arr['Sound Speed'][probe_idx]

    return data_arr

def heat_release_rate_extractor(method, **kwargs):
    ###########################################
    # Internal Functions
    ###########################################
    def cantera_hrr():
        # Step 1: Initialize Cantera solution object
        tmp_obj = ct.Solution(input_params.mech)
        # Step 2: Extract the species composition
        species_name_list = [f"Y({species})" for species in input_params.species]
        tmp_arr, _ = local_data_refinement(raw_data, domain_info, species_name_list, preloaded_grids=grid_arr)
        species_comp = {input_params.species[i]: tmp_arr[f"Y({input_params.species[i]})"][1] for i in range(len(input_params.species))}

        # Step 3: Extract data from preloaded data or from data files
        x_arr, tmp_arr = preload_value_check(['Temperature', 'Pressure'], pre_loaded_data, raw_data, domain_info,
                                             grid_arr, position_flag=True)

        temperature = tmp_arr['Temperature']
        pressure = tmp_arr['Pressure']

        # Step 3: Extract the heat release rate
        tmp_arr = []
        for i in range(len(x_arr)):
            species_values = {key: arr[i] for key, arr in species_comp.items() if len(arr) > i}

            tmp_obj.TPY = (temperature[i],
                           pressure[i],
                           species_values
                           )

            tmp_arr.append(tmp_obj.heat_release_rate)

        return tmp_arr, np.max(tmp_arr)

    ###########################################
    # Main Function
    ###########################################
    # Step 1: Parse input arguments
    raw_data = kwargs.get('raw_data', None)
    grid_arr = kwargs.get('grid_arr', None)
    domain_info = kwargs.get('domain_info', None)
    pre_loaded_data = kwargs.get('pre_loaded_data', None)

    # Step 2:
    if method == 'Cantera':
        return cantera_hrr()
    elif method == 'PeleC':
        x_arr, tmp_arr = preload_value_check(['heatRelease'], pre_loaded_data, raw_data, domain_info,
                                             grid_arr, position_flag=True)
        if tmp_arr['heatRelease'] is None:
            hrr_arr, _ = cantera_hrr()
        else:
            hrr_arr = tmp_arr['heatRelease']

        return hrr_arr, np.max(hrr_arr)
    else:
        raise ValueError('Invalid Method! Must be Cantera or PeleC')

def reynolds_number_extractor(wave_type, **kwargs):
    ###########################################
    # Internal Functions
    ###########################################
    def probe_array():
        wave_loc_adjustments = {
            'Flame': 1e-3,
            'Burned Gas': -1e-3,
            'Maximum Pressure': 0,
            'Pre-Shock': 1e-3,
            'Post-Shock': -1e-3
        }

        if wave_type not in wave_loc_adjustments:
            raise ValueError('Invalid Wave Type!')

        return np.argwhere(x_arr >= (wave_loc + wave_loc_adjustments[wave_type]))[0][0]

    # Step 1: Parse input arguments
    wave_loc = kwargs.get('wave_loc', None)
    raw_data = kwargs.get('raw_data', None)
    grid_arr = kwargs.get('grid_arr', None)
    domain_info = kwargs.get('domain_info', None)
    pre_loaded_data = kwargs.get('pre_loaded_data', None)

    # Step 2: Extract the parameters to calculate reynolds number
    x_arr, tmp_arr = preload_value_check(['Temperature', 'Pressure', 'Density', 'Viscosity', 'Velocity'],
                                         pre_loaded_data, raw_data, domain_info, grid_arr, position_flag=True)

    temperature = tmp_arr['Temperature']
    pressure = tmp_arr['Pressure']
    density = tmp_arr['Density']
    viscosity = tmp_arr['Viscosity']
    velocity = tmp_arr['Velocity']

    # Step 3: Calculate viscosity is nessisary
    if viscosity is None:
        # Step 3.1: Extract the species composition
        species_name_list = [f"Y({species})" for species in input_params.species]
        tmp_arr, _ = local_data_refinement(raw_data, domain_info, species_name_list, preloaded_grids=grid_arr)
        species_comp = {input_params.species[i]: tmp_arr[f"Y({input_params.species[i]})"][1] for i in
                        range(len(input_params.species))}

        tmp_obj = ct.Solution(input_params.mech)
        viscosity = []
        for i in range(len(x_arr)):
            species_values = {key: arr[i] for key, arr in species_comp.items() if len(arr) > i}

            tmp_obj.TPY = (temperature[i],
                           pressure[i],
                           species_values
                           )

            viscosity.append(tmp_obj.viscosity)

    # Step 4: Determine the location of the wave
    try:
        probe_idx = probe_array()
    except Exception as e:
        print(f"Error Could not find thermodynamic location: {e}")
        probe_idx = -1

    # Step 4: Calculate the Reynolds Number
    reynolds_number = ((density * (100 ** 3) / 1000) * (velocity / 100) * (raw_data.domain_width[1].to_value() / 100)) / viscosity

    if wave_loc is not None:
        return reynolds_number[probe_idx], reynolds_number
    else:
        return reynolds_number

############################################################
# Flame Geometry Processing Functions
############################################################

def flame_geometry_function(raw_data, domain_info, output_dir, CHECK_FLAGS):
    ###########################################
    # Internal Functions
    ###########################################
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
        plt.title(raw_data.current_time.to_value())
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        filename = os.path.join(output_dir_path,
                                f"Flame-Length-Animation-{raw_data.basename}.png")
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

        plt.title(f'Flame Normal: {raw_data.current_time.to_value()}')
        filename = os.path.join(output_dir_path,
                                f"Flame-Thickness-Animation-{raw_data.basename}.png")
        plt.savefig(filename, format='png')
        plt.close()

    def sort_by_nearest_neighbors(points, domain_grid):
        buffer = 0.0075 * raw_data.domain_right_edge.to_value()[1]
        valid_indices = (points[:, 1] >= raw_data.domain_left_edge.to_value()[1] + buffer) & (
                points[:, 1] <= raw_data.domain_right_edge.to_value()[1] - buffer)
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

    def flame_thickness(contour_arr, center_val, output_dir_path):
        def extract_simulation_grid():
            # Step 1: Extract the flame location from the contour and simulation array
            flame_idx = np.argmin(abs(contour_arr[:, 1] - center_val[0][1]))
            flame_x, flame_y = contour_arr[flame_idx]

            # Step 2: Collect the max level grids
            max_level = raw_data.index.max_level
            grids = [grid for grid in raw_data.index.grids if grid.Level == max_level]

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
            if min(x_indices, y_indices) < flame_thickness_bin_size:
                subgrid_bin_size = min(x_indices, y_indices)
            else:
                subgrid_bin_size = flame_thickness_bin_size

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
        flame_idx = np.argmin(abs(contour_arr[:, 1] - center_val[0][1]))
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

    def wavelength(output_dir):
        def distance_along_curve(x, y):
            """Compute the distance along the curve."""
            dx = np.gradient(x)
            dy = np.gradient(y)
            ds = np.sqrt(dx ** 2 + dy ** 2)
            s = np.cumsum(ds)
            return s

        def fractal_analysis(s, x, y, scales=128):
            def count_steps(points, caliper_size):
                steps = 0
                i = 0
                while i < len(points) - 1:
                    dist = 0
                    j = i + 1
                    while j < len(points) - 1 and dist < caliper_size:
                        dist = np.linalg.norm(points[j] - points[i])
                        j += 1
                    steps += 1
                    i = j
                return steps

            ###########################################
            # Calculates the fractal dimension of a curve using an overlapping box counting method
            #
            # Args:
            #   x: x-coordinates of the curve
            #   y: y-coordinates of the curve
            #   fractal_bin_size: Size of the bin for the fractal analysis
            #
            # Returns:
            #   fractal_dimension: The fractal dimension of the curve
            ###########################################

            # Convert to numpy arrays
            s = np.array(s)
            x = np.array(x)
            y = np.array(y)

            # Step 1: Determine the bounds on the flame contour
            min_caliper = np.min(np.abs(np.diff(s))[np.abs(np.diff(s)) > 0]) / 2
            max_caliper = s[-1] * 10

            # Step 2: Determine the number of bins in the x and y directions
            """
            # Compute log-space parameters
            mean_log_caliper = (np.log(min_caliper) + np.log(max_caliper)) / 2  # Mean in log space
            std_log_caliper = (np.log(max_caliper) - np.log(min_caliper)) / 4  # Spread in log space
            # Generate log-normal distributed values
            caliper_size = np.random.lognormal(mean=mean_log_caliper, sigma=std_log_caliper, size=10 * scales)
            # Filter values within bounds
            caliper_size = np.sort(caliper_size[(caliper_size >= min_caliper) & (caliper_size <= max_caliper)])
            """
            caliper_size = np.logspace(np.log10(min_caliper), np.log10(max_caliper), scales)

            N_pts = []
            for i in range(len(caliper_size)):
                N_pts.append(count_steps(np.column_stack((x, y)), caliper_size[i]))
            N_pts = np.array(N_pts)

            # Step 3: Determine the inner and outer cuttoff bounds from a smoothed curve

            # Smooth Data
            N_pts_smooth = savgol_filter(N_pts, 10, 1)

            # Step 4:
            window_size = 3
            log_caliper_size = np.log(caliper_size)
            log_N_smooth_pts = np.log(N_pts_smooth)
            log_N_pts = np.log(N_pts)

            left_slope, _, _, _, _ = linregress(log_caliper_size[0:window_size], log_N_smooth_pts[0:window_size])
            right_slope, _, _, _, _ = linregress(log_caliper_size[len(log_caliper_size) - window_size - 1:-1],
                                                 log_N_smooth_pts[len(log_N_smooth_pts) - window_size - 1:-1])

            window_size = 10
            frac_dim_slopes = []
            frac_dim_r_values = []
            slopes_smooth = []
            for i in range(window_size // 2, len(log_caliper_size) - window_size // 2):
                # Select a window of points around the current index 'i'
                x_window = log_caliper_size[i - window_size // 2:i + window_size // 2 + 1]
                y_window = log_N_smooth_pts[i - window_size // 2:i + window_size // 2 + 1]

                # Fit a line (linear regression)
                slope, intercept, r_value, _, _ = linregress(x_window, y_window)
                slopes_smooth.append(slope)

                # Select a window of points around the current index 'i'
                x_window = log_caliper_size[i - window_size // 2:i + window_size // 2 + 1]
                y_window = log_N_pts[i - window_size // 2:i + window_size // 2 + 1]

                # Fit a line (linear regression)
                frac_dim_slope, intercept, r_value, _, _ = linregress(x_window, y_window)
                frac_dim_slopes.append(frac_dim_slope)
                frac_dim_r_values.append(r_value)

            # Step 5: Determine the fractal dimension
            idx = None
            best_r2 = -np.inf  # Start with a very low r²
            best_slope = None

            for i in range(len(frac_dim_slopes)):
                abs_slope = abs(frac_dim_slopes[i])
                if 1 <= abs_slope <= 2:  # Check if the slope is in the expected range
                    current_r2 = frac_dim_r_values[i] ** 2

                    # Ensure it's a stable region by comparing neighboring slopes
                    if i > 0 and i < len(frac_dim_slopes) - 1:
                        slope_variation_pre = abs(frac_dim_slopes[i] - frac_dim_slopes[i - window_size // 2])
                        slope_variation_post = abs(frac_dim_slopes[i] - frac_dim_slopes[i + window_size // 2])
                        if slope_variation_pre > 0.1 and slope_variation_post > 0.1:  # Too much variation indicates instability
                            continue

                    if current_r2 > best_r2:
                        idx = i
                        best_r2 = current_r2
                        best_slope = frac_dim_slopes[i]

            # If no valid slope is found, fall back to the max r² approach
            if idx is None:
                print('No valid slope found. Using the maximum r**2 value.')
                idx = np.argmax(np.array(frac_dim_r_values) ** 2)
                best_slope = frac_dim_slopes[idx]

            # Step 5: Determine the intersection points (inner and outer cutoffs)
            inner_cutoff_line = np.exp(left_slope * log_caliper_size + log_N_smooth_pts[0]) / N_pts[0]
            outer_cutoff_line = np.exp(right_slope * log_caliper_size + log_N_smooth_pts[-1]) / N_pts[0]
            frac_dim_line = np.exp(best_slope * (log_caliper_size - log_caliper_size[idx]) + log_N_smooth_pts[idx]) / \
                            N_pts[
                                0]

            inner_cutoff = caliper_size[np.argmin(abs(inner_cutoff_line - frac_dim_line))]
            outer_cutoff = caliper_size[np.argmin(abs(outer_cutoff_line - frac_dim_line))]

            # Plot the results
            temp_plt_dir = ensure_long_path_prefix(
                os.path.join(output_dir, f"Animation-Frames", f"Wavelength-Plt-Files"))

            if os.path.exists(temp_plt_dir) is False:
                os.makedirs(temp_plt_dir, exist_ok=True)

            fig, axs = plt.subplots(2, 1, figsize=(8, 8))
            # Adjust space between subplots
            fig.subplots_adjust(hspace=0.5)  # Increase spacing between plots

            # Original Data Plot
            axs[0].plot(x, y, 'k-', label="Original Data")
            axs[0].set_xlabel("X (cm)")
            axs[0].set_ylabel("Y (cm)")

            # Fractal Analysis Curve
            axs[1].plot(caliper_size, N_pts / N_pts[0], 'ko', label="Original Data")
            axs[1].plot(caliper_size, N_pts_smooth / N_pts_smooth[0], 'b-', label="Smoothed Data (Savitzky-Golay)")
            axs[1].plot(caliper_size, inner_cutoff_line, 'k--', label="Left Boundary")
            axs[1].plot(caliper_size, outer_cutoff_line, 'k--', label="Right Boundary")
            axs[1].plot(caliper_size, frac_dim_line, 'r-', label="Max Slope")

            # Title with error handling
            try:
                axs[1].set_title(f"Fractal Analysis:\n"
                                 f"Inner Cutoff: {inner_cutoff}, Outer Cutoff: {outer_cutoff}\n"
                                 f"Fractal Dimension (Slope): {abs(best_slope)}",
                                 multialignment='center')
            except Exception as e:
                print(f"Error setting title: {e}")

            axs[1].set_xscale('log')
            axs[1].set_yscale('log')
            axs[1].set_xlabel("Caliper Size")
            axs[1].set_ylabel("N(Caliper Size)")
            axs[1].legend(loc="lower left")

            # Corrected ylim setting
            min_outer = min(outer_cutoff_line)
            max_inner = max(inner_cutoff_line)
            axs[1].set_ylim(min_outer - 0.25 * min_outer, max_inner + 0.25 * max_inner)

            filename = os.path.join(temp_plt_dir, f"Fractal-Animation-{raw_data.basename}.png")
            plt.savefig(filename, format='png')
            plt.close()

            return abs(best_slope)

        # Step 1: Process only the longest segment
        temp_list = []
        for i in range(len(sorted_segments)):
            temp_list.append(len(sorted_segments[i]))

        idx = np.argmax(temp_list)

        # Step 2: Select the points consisting of the flame tip
        x_loc_bnds = np.max(sorted_segments[idx][:, 0]) - 0.25 * (
                    np.max(sorted_segments[idx][:, 0]) - np.min(sorted_segments[idx][:, 0]))

        x_filterd = sorted_segments[idx][:, 0][sorted_segments[idx][:, 0] >= x_loc_bnds]
        y_filterd = sorted_segments[idx][:, 1][sorted_segments[idx][:, 0] >= x_loc_bnds]

        # Step 2: Perform curvature analysis on the longest segment
        curve_dist = distance_along_curve(x_filterd, y_filterd)
        # curvature_analysis(curve_dist, sorted_segments[idx][:, 0], sorted_segments[idx][:, 1])
        fractal_dimension = fractal_analysis(curve_dist, x_filterd, y_filterd, 96)

        return fractal_dimension

    ###########################################
    # Main Function
    ###########################################
    # Step 1:
    domain_grid = domain_info[-1]
    center_val = domain_info[1]

    results = np.empty(3, dtype=object)
    # Step 1: Extract the flame contour and sort the points by nearest neighbors
    raw_data.force_periodicity()
    try:
        try:
            contour_verts = manually_aquire_flame_contour(raw_data)
        except Exception as e:
            contour_verts = raw_data.all_data().extract_isocontours("Temp", flame_temp)
            print(f"Error: Unable to manually extract flame contour: {e}")

        sorted_points, sorted_segments, contour_length = sort_by_nearest_neighbors(contour_verts, domain_grid)

        if 'Domain State Animations' in CHECK_FLAGS:
            if CHECK_FLAGS['Domain State Animations'].get('Surface Contour', False):
                temp_plt_dir = ensure_long_path_prefix(
                    os.path.join(output_dir, f"Animation-Frames", f"Surface-Contour-Plt-Files"))

                if os.path.exists(temp_plt_dir) is False:
                    os.makedirs(temp_plt_dir, exist_ok=True)

                plot_contour(contour_verts, sorted_segments, temp_plt_dir)

    except Exception as e:
        contour_length = np.nan
        print(f"Error: Unable to extract flame contour: {e}")

    # Compute requested metrics
    if CHECK_FLAGS['Flame'].get('Surface Length', False):
        results[0] = contour_length
    if CHECK_FLAGS['Flame'].get('Flame Thickness', False):
        if contour_length != 0:
            try:
                results[1] = flame_thickness(sorted_points, center_val, output_dir)
            except Exception as e:
                results[1] = np.nan
                print(f"Error: Unable to extract flame thickness: {e}")
        else:
            results[1] = np.nan

    if CHECK_FLAGS['Flame'].get('Wavelength', False):
        if contour_length != 0:
            try:
                results[2] = wavelength(output_dir)
            except Exception as e:
                # Get the traceback as a string
                exc_type, exc_value, exc_tb = sys.exc_info()
                traceback_details = traceback.format_exception(exc_type, exc_value, exc_tb)

                # Print the error message with the specific line
                print("".join(traceback_details))

                print(f"Error: Unable to extract flame wavelength: {e}")
                results[2] = np.nan

    return results

########################################################################################################################
# Parallelization Scripts
########################################################################################################################

def single_file_processing(args):
    ###########################################
    # Internal Functions
    ###########################################
    def load_data():
        # Step 1: Load and sort the pelec plot file data
        raw_data = yt.load(pltFile_dir)
        time = raw_data.current_time.to_value()

        # Step 2: Depending on the desired pre-loaded variables
        data, grids = load_plt_data(pltFile_dir, domain_info, CHECK_FLAGS)

        pre_loaded_data = np.empty(len(data) + 2, dtype=object)

        # Initialize the first element as a list
        pre_loaded_data[0] = ['Position']
        pre_loaded_data[1] = data[next(iter(data))][0]

        for i, key in enumerate(data.keys()):
            pre_loaded_data[0].append(key)  # Append the key name
            pre_loaded_data[i + 2] = data[key][1]  # Store corresponding data

        return time, raw_data, pre_loaded_data, grids

    ###########################################
    # Main Function
    ###########################################
    global input_params
    iter_var, const_arr, kwargs = args

    pltFile_dir = iter_var
    domain_info = kwargs.get('domain_info', [])
    output_dir = kwargs.get('output_dir', None)
    animation_bnds = kwargs.get('animation_bnds', None)
    CHECK_FLAGS = kwargs.get('CHECK_FLAGS', [])

    # Step 1: Load the PeleC data from the desired plot file
    time, raw_data, pre_loaded_data, grids = load_data()

    # Step 2: Process the data, extracting the desired parameters for the stated wave types
    result_dict = {'Time': {'Value': time}}

    # Helper function to handle wave tracking and thermodynamic state extraction
    def process_wave(wave_type, result_key):
        result_dict[result_key] = {}
        if CHECK_FLAGS[result_key].get('Position', False):
            if result_key in ['Pre-Shock', 'Post-Shock']:
                result_dict[result_key]['Index'], result_dict[result_key]['Position'] = result_dict['Leading Shock']['Index'], result_dict['Leading Shock']['Position']
            elif result_key in ['Burned Gas']:
                result_dict[result_key]['Index'], result_dict[result_key]['Position'] = result_dict['Flame']['Index'], result_dict['Flame']['Position']
            else:
                try:
                    result_dict[result_key]['Index'], result_dict[result_key]['Position'] = wave_tracking(wave_type,
                                                                                                          raw_data=raw_data,
                                                                                                          grid_arr=grids,
                                                                                                          domain_info=domain_info,
                                                                                                          pre_loaded_data=pre_loaded_data)
                except Exception as e:
                    print(f"Error: Unable to determine {pltFile_dir} {result_key} position: {e}")
                    result_dict[result_key]['Index'] = 0
                    result_dict[result_key]['Position'] = 0

        if CHECK_FLAGS[result_key].get('Thermodynamic State', False):
            if 'Position' not in result_dict[result_key]:
                if result_key in ['Pre-Shock', 'Post-Shock']:
                    result_dict[result_key]['Index'], result_dict[result_key]['Position'] = result_dict['Leading Shock']['Index'], result_dict['Leading Shock']['Position']
                elif result_key in ['Burned Gas']:
                    result_dict[result_key]['Index'], result_dict[result_key]['Position'] = result_dict['Flame']['Index'], result_dict['Flame']['Position']
                else:
                    result_dict[result_key]['Index'], result_dict[result_key]['Position'] = wave_tracking(wave_type,
                                                                                                          raw_data=raw_data,
                                                                                                          grid_arr=grids,
                                                                                                          domain_info=domain_info,
                                                                                                          pre_loaded_data=pre_loaded_data)

            result_dict[result_key]['Thermodynamic State'] = thermodynamic_state_extractor(wave_type,
                                                                                           raw_data=raw_data,
                                                                                           grid_arr=grids,
                                                                                           domain_info=domain_info,
                                                                                           pre_loaded_data=pre_loaded_data,
                                                                                           wave_loc=
                                                                                           result_dict[result_key][
                                                                                               'Position'])
        if CHECK_FLAGS[result_key].get('Reynolds Number', False):
            result_dict[result_key]['Reynolds Number'],  result_dict[result_key]['Reynolds Number Array']= reynolds_number_extractor(wave_type,
                                                                                                                                     raw_data=raw_data,
                                                                                                                                     grid_arr=grids,
                                                                                                                                     domain_info=domain_info,
                                                                                                                                     pre_loaded_data=pre_loaded_data,
                                                                                                                                     wave_loc=result_dict[result_key]['Position'])

    # Step 3: Flame Processing
    if 'Flame' in CHECK_FLAGS:
        process_wave('Flame', 'Flame')
        if CHECK_FLAGS['Flame'].get('Relative Velocity', False):
            temp_var =  next(iter(local_data_refinement(raw_data, domain_info, ['x_velocity'], preloaded_grids=grids)[0].values()))[1]
            result_dict['Flame']['Gas Velocity'] = temp_var[result_dict['Flame']['Index'] + 10]
        if CHECK_FLAGS['Flame'].get('Heat Release Rate Cantera', False):
            result_dict['Flame']['Heat Release Rate Cantera Array'], result_dict['Flame']['Heat Release Rate Cantera'] = heat_release_rate_extractor('Cantera', raw_data=raw_data, grid_arr=grids, domain_info=domain_info)
        if CHECK_FLAGS['Flame'].get('Heat Release Rate PeleC', False):
            try:
                result_dict['Flame']['Heat Release Rate PeleC Array'], result_dict['Flame']['Heat Release Rate PeleC'] = heat_release_rate_extractor('PeleC', raw_data=raw_data, grid_arr=grids, domain_info=domain_info)
            except:
                result_dict['Flame']['Heat Release Rate PeleC Array'], result_dict['Flame']['Heat Release Rate PeleC'] = heat_release_rate_extractor('Cantera', raw_data=raw_data, grid_arr=grids, domain_info=domain_info)
        if CHECK_FLAGS['Flame'].get('Flame Thickness', False) or CHECK_FLAGS['Flame'].get('Surface Length', False) or CHECK_FLAGS['Flame'].get('Wavelength', False):
            result_dict['Flame']['Surface Length'], result_dict['Flame']['Flame Thickness'],  result_dict['Flame']['Wavelength'] = flame_geometry_function(raw_data, domain_info, output_dir, CHECK_FLAGS)

    if 'Burned Gas' in CHECK_FLAGS:
        process_wave('Burned Gas', 'Burned Gas')
        if CHECK_FLAGS['Burned Gas'].get('Velocity', False):
            temp_var = next(iter(local_data_refinement(raw_data, domain_info, ['x_velocity'], preloaded_grids=grids)[0].values()))[1]
            result_dict['Burned Gas']['Velocity'] = temp_var[result_dict['Flame']['Index'] - 10]

    # Step 4: Maximum Pressure Processing
    if 'Maximum Pressure' in CHECK_FLAGS:
        process_wave('Maximum Pressure', 'Maximum Pressure')

    # Step 5: Leading Shock Location Processing
    if 'Leading Shock' in CHECK_FLAGS:
        process_wave('Leading Shock', 'Leading Shock')

    # Step 6: Pre-Shock Processing
    if 'Pre-Shock' in CHECK_FLAGS:
        process_wave('Pre-Shock', 'Pre-Shock')

    # Step 7: Post-Shock Processing
    if 'Post-Shock' in CHECK_FLAGS:
        process_wave('Post-Shock', 'Post-Shock')

    # Step 8: Domain State Animations
    if 'Domain State Animations' in CHECK_FLAGS:
        for key, value in CHECK_FLAGS.get('Domain State Animations', {}).items():
            # Skip specific keys
            if key in ['Surface Contour', 'Flame Thickness', 'Wavelength']:
                continue

            # Check if value is a dictionary and extract relevant flags
            if isinstance(value, dict):
                bool_check = value.get('Flag', False)
                pele_name = value.get('PeleC', key)
            else:
                continue

            if bool_check:
                # Handle x_data_arr
                x_data_arr, tmp_arr = preload_value_check([key], pre_loaded_data, raw_data, domain_info, grids,
                                                          position_flag=True)
                if key in ['Heat Release Rate Cantera', 'Heat Release Rate PeleC', 'Reynolds Number']:
                    y_data_arr = result_dict['Flame'].get(f'{key} Array', None)
                else:
                    y_data_arr = tmp_arr[key]

                # Determine bounds
                bnd_arr_index = [item[0] for item in animation_bnds].index(key)
                y_lim = [animation_bnds[bnd_arr_index][1], animation_bnds[bnd_arr_index][2]]

                # Prepare directory for output
                if " " in key:  # Check if there's a space
                    animation_str = key.replace(" ", "-")  # Join with a hyphen
                else:
                    animation_str = key  # If no space, keep the original string

                temp_plt_dir = ensure_long_path_prefix(os.path.join(output_dir, "Animation-Frames", f"{animation_str}-Plt-Files"))
                os.makedirs(temp_plt_dir, exist_ok=True)

                # Call state_animation
                state_animation(
                    method='Plot',
                    time=time,
                    x_data_arr=x_data_arr,
                    y_data_arr=y_data_arr,
                    y_bounds=y_lim,
                    domain_size=domain_info,
                    var_name=key,
                    plt_folder=raw_data.basename,
                    output_dir_path=temp_plt_dir
                )

    if 'Combined State Animations' in CHECK_FLAGS:
        # Step 1: Extract the names of the enabled subdictionaries
        subdict_names = [
            key for key, value in CHECK_FLAGS.get('Combined State Animations', {}).items()
            if isinstance(value, dict) and value.get('Flag', False)
        ]
        animation_str = []
        for name in subdict_names:
            if " " in name:  # Check if there's a space
                animation_str.append(name.replace(" ", "-"))  # Join with a hyphen
            else:
                animation_str.append(name)  # If no space, keep the original string
        # Step 2: Extract the pelec strings of the enabled subdictionaries

        state_animations = CHECK_FLAGS.get('Combined State Animations', {})
        valid_keys = {
            key: value for key, value in state_animations.items()
            if isinstance(value, dict) and value.get('Flag', False)
        }

        i = 0
        y_data_arr = np.empty(len(valid_keys.keys()), dtype=object)
        for key, value in valid_keys.items():
            # Handle x_data_arr
            x_data_arr, tmp_arr = preload_value_check([key], pre_loaded_data, raw_data, domain_info, grids,
                                                      position_flag=True)
            if key in ['Heat Release Rate Cantera', 'Heat Release Rate PeleC', 'Reynolds Number']:
                y_data_arr[i] = result_dict['Flame'].get(f'{key} Array', None)
            else:
                y_data_arr[i] = tmp_arr[key]

            i += 1

        # Step 4: Find the corresponding bounds for each subdictionary
        bnd_arr_indices = [
            [item[0] for item in animation_bnds].index(name) for name in subdict_names
        ]
        y_lims = [[animation_bnds[idx][1], animation_bnds[idx][2]] for idx in bnd_arr_indices]

        # Step 5: Create the output directory for the combined state animations
        temp_plt_dir = ensure_long_path_prefix(os.path.join(output_dir, f"Animation-Frames", f"{'-'.join(animation_str)}-Plt-Files"))
        os.makedirs(temp_plt_dir, exist_ok=True)

        state_animation(method='Plot',
                        time=time,
                        x_data_arr=x_data_arr,
                        y_data_arr=y_data_arr,
                        y_bounds=y_lims,
                        domain_size=domain_info,
                        var_name=subdict_names,
                        plt_folder=raw_data.basename,
                        output_dir_path=temp_plt_dir)

    # Step 9: Local State Animation
    if 'Local State Animations' in CHECK_FLAGS:
        # Step 1: Extract the names of the enabled subdictionaries
        subdict_names = [
            key for key, value in CHECK_FLAGS.get('Local State Animations', {}).items()
            if isinstance(value, dict) and value.get('Flag', False)
        ]
        animation_str = []
        for name in subdict_names:
            if " " in name:  # Check if there's a space
                animation_str.append(name.replace(" ", "-"))  # Join with a hyphen
            else:
                animation_str.append(name)  # If no space, keep the original string

        # Step 2: Extract the strings of the enabled subdictionaries
        state_animations = CHECK_FLAGS.get('Combined State Animations', {})
        valid_keys = {
            key: value for key, value in state_animations.items()
            if isinstance(value, dict) and value.get('Flag', False)
        }

        # Step 3: Create a temporary array to store the plot data for each enabled subdictionary
        i = 0
        y_data_arr = np.empty(len(valid_keys.keys()), dtype=object)
        for key, value in valid_keys.items():
            # Handle x_data_arr
            x_data_arr, tmp_arr = preload_value_check([key], pre_loaded_data, raw_data, domain_info, grids,
                                                      position_flag=True)
            if key in ['Heat Release Rate Cantera', 'Heat Release Rate PeleC', 'Reynolds Number']:
                y_data_arr[i] = result_dict['Flame'].get(f'{key} Array', None)
            else:
                y_data_arr[i] = tmp_arr[key]

            i += 1

        # Step 4: Find the corresponding bounds for each subdictionary
        bnd_arr_indices = [
            [item[0] for item in animation_bnds].index(name) for name in subdict_names
        ]

        local_physical_window = CHECK_FLAGS['Local State Animations']['Physical Window']
        wave_loc = result_dict[f'{CHECK_FLAGS['Local State Animations']['Wave of Interest']}']['Position'] if 'Position' in result_dict[f'{CHECK_FLAGS['Local State Animations']['Wave of Interest']}'] else wave_tracking(raw_data=raw_data, grid_arr=grids, domain_info=domain_info, pre_loaded_data=pre_loaded_data)

        x_idx = [np.searchsorted(x_data_arr, wave_loc - local_physical_window, side='left'),
                 np.searchsorted(x_data_arr, wave_loc + local_physical_window, side='right')]
        x_lim = [x_data_arr[x_idx[0]], x_data_arr[x_idx[1]]]
        y_lims = [[animation_bnds[idx][1], animation_bnds[idx][2]] for idx in bnd_arr_indices]

        # Step 5: Create the output directory for the combined state animations
        temp_plt_dir = ensure_long_path_prefix(
            os.path.join(output_dir, f"Animation-Frames", f"Local-{'-'.join(animation_str)}-Plt-Files"))
        os.makedirs(temp_plt_dir, exist_ok=True)

        state_animation(method='Plot',
                        time=time,
                        x_data_arr=x_data_arr,
                        y_data_arr=y_data_arr,
                        x_bounds=x_lim,
                        y_bounds=None,
                        domain_size=domain_info,
                        var_name=subdict_names,
                        plt_folder=raw_data.basename,
                        output_dir_path=temp_plt_dir)

    return result_dict

def pelec_processing(pelec_dirs, domain_info, animation_bnds, output_dir, CHECK_FLAGS, SMOOTHING_FLAG):
    ###########################################
    # Internal Functions
    ###########################################
    def file_output(file_path, smoothing_check):
        # Step 1: Dynamically create the text file header depending on the assigned flags
        header_data = ["Time [s]"]

        # Step 2: Loop over processing objectives (flame, Leading shock, maximum pressure, pre-shock, post-shock)
        for key in CHECK_FLAGS.keys():
            if key in {'Flame', 'Burned Gas', 'Leading Shock', 'Maximum Pressure', 'Pre-Shock', 'Post-Shock'}:
                sub_dict = CHECK_FLAGS.get(key, {})
                for sub_key, sub_value in sub_dict.items():
                    if sub_value:
                        if sub_key == 'Thermodynamic State':
                            header_data.extend([f"{key} Temperature [K]", f"{key} Pressure [Pa]",
                                                f"{key} Density [kg/m^3]", f"{key} Soundspeed [m/s]"])
                        else:
                            unit_str = {
                                'Position': 'm', 'Flame Thickness': 'm', 'Surface Length': 'm',
                                'Velocity': 'm/s', 'Relative Velocity': 'm/s',
                                'Heat Release Rate Cantera': 'W/m^3', 'Heat Release Rate PeleC': 'erg/cm3'
                            }.get(sub_key, '[]')
                            header_data.extend([f"{key} {sub_key} [{unit_str}]"])

        # Step 3:
        with open(file_path, "w") as outfile:
            outfile.write("#" + " ".join(f"{i + 1:<55.0f}" for i in range(len(header_data))) + "\n#")
            outfile.write(" ".join(f"{header:<55s}" for header in header_data) + "\n")

            # Step 3: Write Data
            time_key = 'Value'
            for i in range(len(collective_results['Time'][time_key])):
                # Write Time
                outfile.write(f" {collective_results['Time'][time_key][i]:<55e}")
                # Write Other Data
                for key in ('Flame', 'Burned Gas', 'Leading Shock', 'Maximum Pressure', 'Pre-Shock', 'Post-Shock'):
                    sub_dict = CHECK_FLAGS.get(key, {})
                    for sub_key, sub_value in sub_dict.items():
                        if sub_value:
                            if sub_key == 'Thermodynamic State':
                                types = ("Temperature", "Pressure", "Density", "Soundspeed")
                                temp_val = np.array([convert_units(value, t) for value, t in
                                                     zip(collective_results[key][sub_key][i], types)])
                                outfile.write(" ".join(f"{temp_val[j]:<55e}" for j in range(len(temp_val))))
                            else:
                                temp_val = convert_units(collective_results[key][sub_key][i], sub_key)
                                outfile.write(f" {temp_val:<55e}")
                outfile.write("\n")

    ###########################################
    # Main Function
    ###########################################
    # Step 1: Process the individual pelec data files in parallel using multiprocessing
    pelec_data = parallel_processing_function(pelec_dirs, (), single_file_processing,
                                              domain_info=domain_info,
                                              CHECK_FLAGS=CHECK_FLAGS,
                                              animation_bnds=animation_bnds,
                                              output_dir=output_dir)

    # Step 2: Re-organize the processed data for ease of manipulation
    collective_results = {master_key: {slave_key: [pelec_data[i][master_key][slave_key] for i in range(len(pelec_data))] for slave_key in pelec_data[0][master_key]} for master_key in pelec_data[0]}

    # Step 3: Process each key in CHECK_FLAGS
    def process_key(key):
        if key in CHECK_FLAGS:
            print(f'Starting {key} Processing')
            if CHECK_FLAGS[key].get('Velocity', False):
                collective_results[key]['Velocity'] = np.gradient(collective_results[key]['Position']) / np.gradient(collective_results['Time']['Value'])
            if CHECK_FLAGS[key].get('Relative Velocity', False):
                if 'Velocity' in collective_results[key]:
                    collective_results[key]['Relative Velocity'] = collective_results[key]['Velocity'] - collective_results[key]['Gas Velocity']
                else:
                    print('ERROR: Must Enable Velocity Flag to compute the relative velocity')
            print(f'Completed {key} Processing')

    for key in ['Flame', 'Leading Shock', 'Pre-Shock', 'Post-Shock']:
        process_key(key)

    # Step 8: Write to file, if any of the sub-dictionary values except 'Domain State Animations' are true
    print('Start Output File Writing')
    write_to_file = any(any(sub_dict.values()) for key, sub_dict in CHECK_FLAGS.items() if key != 'Domain State Animations')

    if write_to_file:
        file_output(ensure_long_path_prefix(os.path.join(output_dir, f'Wave-Tracking-Results-V{version}.txt')), False)
    if SMOOTHING_FLAG:
        data_smoothing_function(version)
    print('Completed Output File Writing')

    # Step 7: Create Variable Evolution
    print('Starting Animation Processing')
    def process_animations(animation_type, prefix=None):
        subdict_names = [
            key for key, value in CHECK_FLAGS.get(animation_type, {}).items()
            if isinstance(value, dict) and value.get('Flag', False)
        ]

        if animation_type == 'Domain State Animations':
            for name in subdict_names:
                if " " in name:  # Check if there's a space
                    animation_str = name.replace(" ", "-")  # Join with a hyphen
                else:
                    animation_str = name  # If no space, keep the original string

                temp_plt_dir = ensure_long_path_prefix(os.path.join(output_dir, f"Animation-Frames", f"{animation_str}-Plt-Files"))
                animation_filename = ensure_long_path_prefix(os.path.join(output_dir, f"{animation_str}-Evolution-Animation.mp4"))

                try:
                    state_animation(
                        method='Animate',
                        folder_path=temp_plt_dir,
                        animation_filename=animation_filename,
                    )
                except Exception as e:
                    print(f"Error: Unable to create {animation_str} animation: {e}")
        else:
            animation_str = []
            for name in subdict_names:
                if " " in name:  # Check if there's a space
                    animation_str.append(name.replace(" ", "-"))  # Join with a hyphen
                else:
                    animation_str.append(name)  # If no space, keep the original string

            animation_str = '-'.join(animation_str)

            if prefix is None:
                temp_plt_dir = ensure_long_path_prefix(os.path.join(output_dir, f"Animation-Frames", f"{animation_str}-Plt-Files"))
                animation_filename = ensure_long_path_prefix(os.path.join(output_dir, f"{animation_str}-Evolution-Animation.mp4"))
            else:
                temp_plt_dir = ensure_long_path_prefix(os.path.join(output_dir, f"Animation-Frames", f"{prefix}-{animation_str}-Plt-Files"))
                animation_filename = ensure_long_path_prefix(os.path.join(output_dir, f"{prefix}-{animation_str}-Evolution-Animation.mp4"))

            try:
                state_animation(
                    method='Animate',
                    folder_path=temp_plt_dir,
                    animation_filename=animation_filename,
                )
            except Exception as e:
                print(f"Error: Unable to create {animation_str} animation: {e}")

    process_animations('Domain State Animations', )
    process_animations('Combined State Animations', )
    process_animations('Local State Animations', prefix=f"Local")

    print('Completed Animation Processing')
    return

########################################################################################################################
# Main Script
########################################################################################################################

def main():
    start_time = time.time()
    ####################################################################################################################
    # This code is developed to process a 2D Planar Flame simulated using PeleC for a given y-position and a given
    # (temperature) isotherm for the flame and pressure for any shock
    #
    # All functions are configured for a 2 dimensional space
    ####################################################################################################################
    # Step 1: Set all the desired tasks to be performed by the python script
    #row_idx = 'DDT'
    row_idx = 0.0462731
    ddt_dir = '../../../Domain-Length-284cm/0.09cm-Complete-Domain/Planar-Kernel-Level-6-Part-3'
    ddt_plt_file = 'plt332330'

    SMOOTHING_FLAG = True
    CHECK_FLAGS = {
        'Flame': {
            'Position': True,
            'Velocity': True,
            'Relative Velocity': True,
            'Thermodynamic State': True,
            'Heat Release Rate Cantera': True,
            'Heat Release Rate PeleC': True,
            'Flame Thickness': True,
            'Surface Length': True,
            'Wavelength': True,
            'Reynolds Number': True
        },
        'Burned Gas': {
            'Velocity': True,
            'Thermodynamic State': True
        },
        'Leading Shock': {
            'Position': True,
            'Velocity': True
        },
        'Maximum Pressure': {
            'Position': True,
            'Thermodynamic State': True
        },
        'Pre-Shock': {
            'Thermodynamic State': True
        },
        'Post-Shock': {
            'Thermodynamic State': True
        },
        'Domain State Animations': {
            'Temperature': {'PeleC':'Temp', 'Preload': 'Temperature', 'Flag':True},
            'Pressure': {'PeleC':'pressure', 'Preload': 'Pressure', 'Flag':True},
            'Velocity': {'PeleC':'x_velocity', 'Preload': 'Velocity', 'Flag':True},
            'Species': {'PeleC':None, 'Preload': None, 'Flag':None},
            'Heat Release Rate Cantera': {'PeleC':None, 'Preload': None, 'Flag':True},
            'Heat Release Rate PeleC':   {'PeleC':'heatRelease', 'Preload': 'Heat Release Rate', 'Flag':True},
            'Surface Contour': {'PeleC':None, 'Preload': None, 'Flag':True},
            'Flame Thickness': {'PeleC':None, 'Preload': None, 'Flag':True},
            'Wavelength': {'PeleC':None, 'Preload': None, 'Flag':True},
            'Reynolds Number': {'PeleC': None, 'Preload': None, 'Flag': True}
        },
        'Combined State Animations': {
            'Temperature': {'PeleC':'Temp', 'Preload': 'Temperature', 'Flag':True, 'Local': False},
            'Pressure': {'PeleC':'pressure', 'Preload': 'Pressure', 'Flag':True, 'Local': False},
            'Velocity': {'PeleC':'x_velocity', 'Preload': 'Velocity', 'Flag':False, 'Local': False},
            'Species': False,
            'Heat Release Rate Cantera': False,
            'Heat Release Rate PeleC': {'PeleC':'heatRelease', 'Preload': 'Heat Release Rate', 'Flag':True, 'Local': False}
        },
        'Local State Animations':{
            'Wave of Interest': 'Flame',
            'Physical Window': 0.01,
            'Temperature': {'PeleC': 'Temp', 'Preload': 'Temperature', 'Flag': True},
            'Pressure': {'PeleC': 'pressure', 'Preload': 'Pressure', 'Flag': True},
            'Velocity': {'PeleC': 'x_velocity', 'Preload': 'Velocity', 'Flag': False},
            'Species': False,
            'Heat Release Rate Cantera': False,
            'Heat Release Rate PeleC': {'PeleC': 'heatRelease', 'Preload': 'Heat Release Rate', 'Flag': True}
        }
    }

    # Step 2: Initialize the code with the desired processed variables and mixture composition
    initialize_parameters(
        T=503.15,
        P=10.0 * 100000,
        Phi=1.0,
        Fuel='H2',
        mech='Li-Dryer-H2-mechanism.yaml',
    )

    # Step 3: Collect all the present pelec data directories
    dir_path = os.path.dirname(os.path.realpath(__file__))

    time_data_dir = [os.path.join(dir_path, raw_data_folder, time_step)
                     for raw_data_folder in os.listdir(dir_path)
                     if os.path.isdir(os.path.join(dir_path, raw_data_folder)) and raw_data_folder.startswith(f'Raw-{data_set}')
                     for time_step in os.listdir(os.path.join(dir_path, raw_data_folder))
                     if os.path.isdir(os.path.join(dir_path, raw_data_folder, time_step)) and time_step.startswith('plt')]

    # Step 4: Chronologically order the pltFiles and truncate the raw data list if skip loading is enabled
    updated_data_list = sort_files(time_data_dir)

    # Step 5: Determine the domain sizing parameters (size, # of cells)
    domain_info = domain_size_parameters(
        os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), ddt_dir, f'Raw-{data_set}-Data', ddt_plt_file)) if row_idx == 'DDT' else updated_data_list[0], row_idx)

    # Step 6: Create the result directories
    os.makedirs(os.path.join(dir_path, f"Processed-Global-Results-V{version}", f"y-{domain_info[1][0][1]:.3g}cm"), exist_ok=True)
    output_dir_path = os.path.join(dir_path, f"Processed-Global-Results-V{version}", f"y-{domain_info[1][0][1]:.3g}cm")

    # Step 7:
    animation_axis_bnds = animation_axis(updated_data_list, ddt_dir, ddt_plt_file, domain_info, CHECK_FLAGS)

    # Step 8:
    print('Beginning PeleC Processing')
    pelec_processing(updated_data_list, domain_info, animation_axis_bnds, output_dir_path, CHECK_FLAGS, SMOOTHING_FLAG)
    print('Completed PeleC Processing')

    end_time = time.time()
    print(f"Execution time: {end_time - start_time} seconds")
    return

if __name__ == '__main__':
    main()
