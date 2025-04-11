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
import matplotlib.ticker as ticker
import cantera as ct
import numpy as np
import pandas as pd

from functools import reduce

yt.set_log_level(0)

########################################################################################################################
# Global Program Setting Variables
########################################################################################################################
version = 35

flame_thickness_bin_size = 11
flame_temp = 2500
plotting_bnds_bin = 0
n_procs = 1

data_set = 'Raw-PeleC'

########################################################################################################################
# Global Classes
########################################################################################################################

#################################################################
# Initial Thermodynamic Condition Class
#################################################################
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

#################################################################
# Unit Conversion Class
#################################################################

# Global instance of the Pele Variable Map
PELE_VAR_MAP = {
        'X': {'Name': 'x','Units': 'cm'},
        'Y': {'Name': 'y', 'Units': 'cm'},
        'Temperature': {'Name': 'Temp', 'Units': 'K'},
        'Pressure': {'Name': 'pressure', 'Units': 'g / cm / s^2'},
        'Density': {'Name': 'density', 'Units': 'g / cm^3'},
        'Viscosity': {'Name': 'viscosity', 'Units': 'g / cm / s'},
        'Conductivity': {'Name': 'conductivity', 'Units': 'g cm^2 / s^3 / cm / K'},
        'Sound speed': {'Name': 'soundspeed', 'Units': 'cm / s'},
        'Mach Number': {'Name': 'MachNumber', 'Units': 'cm'},
        'X Velocity': {'Name': 'x_velocity', 'Units': 'cm / s'},
        'Y Velocity': {'Name': 'y_velocity', 'Units': 'cm / s'},
        'Heat Release Rate': {'Name': 'heatRelease', 'Units': 'g cm^2 / s^3 / cm^3'},
        'Cp': {'Name': 'cp', 'Units': 'g cm^2 / s^2 / g / K'},
        'Cv': {'Name': 'cv', 'Units': 'g cm^2 / s^2 / g / K'},
    }

# Adding species variables dynamically
def add_species_vars(species_list):
    for species in species_list:
        PELE_VAR_MAP[f'Y({species})'] = ''
        PELE_VAR_MAP[f'D({species})'] = ''
        PELE_VAR_MAP[f'W({species})'] = 'g / cm^3'


class UnitConverter:
    UNIT_MAP = {
        's': 1,
        'g': 1e-3,
        'mol': 1e-3,
        'cm': 1e-2,
        'K': 1,
        'kg': 1,
        'm': 1,
    }

    @classmethod
    def convert(cls, value: float, var_name: str) -> float:
        """
        Convert a value from CGS to MKS based on a variable name.

        Args:
            value (float): The numerical value in CGS.
            var_name (str): The variable name (e.g., "Pressure").

        Returns:
            float: The converted value in MKS.
        """
        # Look up the unit expression for the given variable
        unit_expr = cls._lookup_unit_expr(var_name)

        # Split the expression into numerator and denominator
        num_units, denom_units = cls._parse_units(unit_expr)

        # Compute conversion factor
        num_factor = cls._convert_units(num_units, exponent=1)
        denom_factor = cls._convert_units(denom_units, exponent=-1) if denom_units else 1

        return value * num_factor * denom_factor

    @classmethod
    def _lookup_unit_expr(cls, var_name: str) -> str:
        if var_name not in PELE_VAR_MAP:
            raise ValueError(f"Unknown variable '{var_name}'. Available variables: {list(PELE_VAR_MAP.keys())}")

        unit_info = PELE_VAR_MAP[var_name]

        if isinstance(unit_info, dict) and 'Units' in unit_info:
            return unit_info['Units']  # Ensure it returns a string
        elif isinstance(unit_info, str) and unit_info:
            return unit_info  # Handle cases where units were added as strings (e.g., species)
        else:
            raise ValueError(f"Variable '{var_name}' does not have a defined unit.")

    @classmethod
    def _parse_units(cls, unit_expr: str) -> tuple:
        """
        Parses a compound unit expression into numerator and denominator parts.

        Args:
            unit_expr (str): The unit string (e.g., "g cm^2 / s^3 / cm / K").

        Returns:
            tuple: (numerator_units, denominator_units) as lists of terms.
        """
        terms = re.split(r'\s*/\s*', unit_expr)  # Split by '/' while allowing spaces
        num_units = terms[0].split()  # First section is the numerator
        denom_units = [] if len(terms) == 1 else [t for term in terms[1:] for t in term.split()]  # Flatten denominators

        return num_units, denom_units

    @classmethod
    def _convert_units(cls, units: list, exponent: int) -> float:
        """
        Convert units in a numerator or denominator, applying exponent adjustments.

        Args:
            units (list): List of unit terms (e.g., ["g", "cm^2", "s^3"]).
            exponent (int): +1 for numerator, -1 for denominator.

        Returns:
            float: The conversion factor for the given units.
        """
        if not units:
            return 1  # No units means neutral factor

        factors = []
        for unit in units:
            match = re.match(r"([a-zA-Z]+)(?:\^(-?\d+))?", unit)  # Extract base unit & exponent
            if not match:
                raise ValueError(f"Invalid unit format: {unit}")

            base_unit, exp = match.group(1), match.group(2)
            if base_unit not in cls.UNIT_MAP:
                raise ValueError(f"Unknown unit '{base_unit}'. Available units: {list(cls.UNIT_MAP.keys())}")

            base_factor = cls.UNIT_MAP[base_unit]
            power = int(exp) if exp else 1  # Default exponent is 1
            factors.append(base_factor ** (power * exponent))

        return reduce(lambda x, y: x * y, factors, 1)  # Multiply all factors together

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
# Function Scripts - Data Importing
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
    x_coords = np.arange(ds.domain_left_edge[0].to_value() / 100, ds.domain_right_edge[0].to_value() / 100, data.dds[0].to_value() / 100)
    y_coords = np.arange(ds.domain_left_edge[1].to_value() / 100, ds.domain_right_edge[1].to_value() / 100, data.dds[1].to_value() / 100)
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
        y_slice_index = np.argmin(abs(y_arr - (np.max(y_arr) + np.min(y_arr)) / 2))
        y_slice_loc = y_arr[y_slice_index]

    return (np.array([[0, y_slice_index], [data.ActiveDimensions[0], y_slice_index]]),
            np.array([[data.LeftEdge[0].to_value() / 100, y_slice_loc / 100], [data.RightEdge[0].to_value() / 100, y_slice_loc / 100]]),
            grid_arr)

def pelec_data_extraction(raw_data, extract_location, direction='x', preloaded_grids=None):
    ##########################################
    # Internal Functions
    ###########################################
    def cantera_str_acquisition(grid, idx, gas):

        # Step 1: Extract the thermodynamic state and composition
        T = grid["boxlib", "Temp"].to_value().flatten()[idx]
        P = grid["boxlib", "pressure"].to_value().flatten()[idx] / 10

        species_list = [var for var in input_params.species]
        Y = {
            species: grid["boxlib", f"Y({species})"].to_value().flatten()[i]  # Construct key dynamically
            for species in species_list if f"Y({species})" in np.array(raw_data.field_list)[:, 1]  # Ensure key exists
        }

        # Step 2: Modify the gas object
        gas.TPY = T, P, Y

        # Step 3:
        missing_data = {}
        for key, var in missing_str.items():
            if key == 'Density':
                missing_data[key] = gas.density_mass
            if key == 'Viscosity':
                missing_data[key] = gas.viscosity
            if key == 'Conductivity':
                missing_data[key] = gas.thermal_conductivity
            if key == 'Sound speed':
                missing_data[key] = gas.sound_speed
            if key == 'Mach Number':
                missing_data[key] = grid["boxlib", "x_velocity"].to_value().flatten()[idx] / gas.sound_speed
            if key == 'Heat Release Rate':
                missing_data[key] = gas.heat_release_rate
            if key == 'Cp':
                missing_data[key] = gas.cp_mass
            if key == 'Cv':
                missing_data[key] = gas.cv_mass
            if key == 'rho_e':
                missing_data[key] = gas.density_mass * gas.int_energy_mass

            for species in input_params.species:
                if key == f'rho_{species}':
                    missing_data[key] = gas.density_mass * gas.Y[gas.species_index(species)]
                if key == f'Y({species})':
                    missing_data[key] = gas.Y[gas.species_index(species)]
                if key == f'D({species})':
                    missing_data[key] = gas.mix_diff_coeffs_mass[gas.species_index(species)]
                if key == f'h({species})':
                    missing_data[key] = gas.standard_enthalpies_RT[gas.species_index(species)] * ct.gas_constant * gas.T
                if key == f'W({species})':
                    missing_data[key] = gas.net_production_rates[gas.species_index(species)] * gas.molecular_weights[gas.species_index(species)]

        return missing_data

    ##########################################
    # Main Function
    ###########################################
    """
        Extracts refined data for multiple required variables.

        Parameters:
            raw_data: The full simulation dataset.
            extract_location: Information about the domain (e.g., y-level for extraction).
            required_vars: A list of variable names that need to be loaded.
            preloaded_grids: Optional dictionary of preloaded grids for each refinement level.

        Returns:
            A dictionary of {mapped_variable_name: (x_sorted, var_sorted)}
    """
    PELE_VAR_MAP = {
        'X': {'Name': 'x','Units': 'cm'},
        'Y': {'Name': 'y', 'Units': 'cm'},
        'Temperature': {'Name': 'Temp', 'Units': 'K'},
        'Pressure': {'Name': 'pressure', 'Units': 'g / cm / s^2'},
        'Density': {'Name': 'density', 'Units': 'g / cm^3'},
        'Viscosity': {'Name': 'viscosity', 'Units': 'g / cmm / s'},
        'Conductivity': {'Name': 'conductivity', 'Units': 'g cm^2 / s^3 / cm / K'},
        'Sound speed': {'Name': 'soundspeed', 'Units': 'cm / s'},
        'Mach Number': {'Name': 'MachNumber', 'Units': 'cm'},
        'X Velocity': {'Name': 'x_velocity', 'Units': 'cm / s'},
        'Y Velocity': {'Name': 'y_velocity', 'Units': 'cm / s'},
        'Heat Release Rate': {'Name': 'heatRelease', 'Units': 'g cm^2 / s^3 / cm^3'},
        'Cp': {'Name': 'cp', 'Units': 'g cm^2 / s^2 / g / K'},
        'Cv': {'Name': 'cv', 'Units': 'g cm^2 / s^2 / g / K'},
    }

    for species in input_params.species:
        PELE_VAR_MAP[f'Y({species})'] = {'Name': f'Y({species})', 'Units': ''}
        PELE_VAR_MAP[f'D({species})'] = {'Name': f'D({species})', 'Units': ''}
        PELE_VAR_MAP[f'W({species})'] = {'Name': f'rho_omega_{species}', 'Units': 'g / cm^3'}

    # Check that all the target strings are in the data file, if not then use cantera to fill the gaps
    missing_str = {
        key: raw_var["Name"]
        for key, raw_var in PELE_VAR_MAP.items()
        if raw_var["Name"] not in np.array(raw_data.field_list)[:, 1] and key != "Grid" and key != 'X' and key != 'Y'
    }
    if missing_str:
        gas_missing = ct.Solution(input_params.mech)

    # Get the maximum refinement level and smallest grid spacing
    max_level = raw_data.index.max_level
    dx_min = raw_data.index.get_smallest_dx()
    # Dictionary to store refined data per level
    # Create level_data using the 'Name' field from PELE_VAR_MAP
    level_data = {level: {PELE_VAR_MAP[var]["Name"]: {} for var in PELE_VAR_MAP.keys()} for level in
                  range(max_level + 1)}
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
            if direction == 'y':
                # dx = dx_min.to_value()
                mask = np.isclose(x, extract_location * 100, atol=dx)  # Allow small tolerance
            else:
                mask = np.isclose(y, extract_location * 100, atol=dx)  # Allow small tolerance
            if np.any(mask) and not preloaded_grids:
                used_grids.setdefault(level, []).append(grid)  # Store grids only if not preloaded

            # Process each required variable
            for var in PELE_VAR_MAP.keys():
                var_name = PELE_VAR_MAP[var]["Name"]  # Get mapped name
                if direction == 'y':
                    position_arr = y
                else:
                    position_arr = x

                if var in missing_str.keys():  # Only compute if missing_str is not empty
                    for i, xi in enumerate(position_arr[mask]):
                        temp_var = cantera_str_acquisition(grid, i, gas_missing)
                        level_data[level][var_name][xi] = temp_var[var]  # Higher levels overwrite lower ones
                else:
                    try:
                        temp_var = grid["boxlib", var_name].flatten()  # Use the mapped name for lookup
                        for xi, vi in zip(position_arr[mask], temp_var[mask]):
                            level_data[level][var_name][xi] = vi  # Store using mapped name
                    except:
                        continue

    # Convert to sorted NumPy arrays and apply variable name mapping for each level
    final_data = {
        level: {
            var: (  # Use the original key as the dictionary key
                np.array(sorted(level_data[level].get(PELE_VAR_MAP[var]["Name"], {}).keys()))
                if PELE_VAR_MAP[var]["Name"] in level_data[level] else np.array([]),

                np.array([
                    level_data[level].get(PELE_VAR_MAP[var]["Name"], {}).get(xi, 0)
                    for xi in sorted(level_data[level].get(PELE_VAR_MAP[var]["Name"], {}).keys())
                ])
            )
            for var in PELE_VAR_MAP  # Iterate over the original dictionary keys
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
                continue  # Skip empty data

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

        # Now apply the unit conversion for each variable in the merged data
        unit_expr = PELE_VAR_MAP.get(var, {}).get('Units', '')
        if unit_expr:
            # Apply the unit conversion using the UnitConverter class
            converted_y = [UnitConverter.convert(yi, var) for yi in merged_y]
            merged_data[var] = np.array(converted_y)
        else:
            merged_data[var] = merged_y

    return merged_data, used_grids

########################################################################################################################
# Function Scripts - Data Processing
########################################################################################################################

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
        elif wave_type == 'Lead Shock':
            return np.argwhere(data_arr['Pressure'] >= 1.01 * data_arr['Pressure'][-1])[-1][0]
        else:
            raise ValueError('Invalid Wave Type! Must be Flame, Maximum Pressure, or Leading Shock')

    ###########################################
    # Main Function
    ###########################################
    # Step 1: Parse input arguments
    pre_loaded_data = kwargs.get('pre_loaded_data')

    # Step 2: Define wave type and data string relations
    WAVE_TYPE_MAP = {
        'Flame': {
            'PeleC': ('Temp', 'Y(HO2)'),
            'Pre Loaded': ('Temperature', 'Y(HO2)')
        },
        'Lead Shock': {
            'PeleC': 'pressure',
            'Pre Loaded': 'Pressure'
        }
    }

    data_str = WAVE_TYPE_MAP[wave_type]['Pre Loaded'] if pre_loaded_data is not None else WAVE_TYPE_MAP[wave_type]['PeleC']

    # Step 3: Extract the data for the wave type
    x_arr = pre_loaded_data['X']
    data_arr = pre_loaded_data

    # Step 4: Find the index of the wave
    wave_idx = find_wave_index(wave_type, data_str)

    return wave_idx, x_arr[wave_idx]

def boundary_layer_extractor(raw_data, flame_location, shock_location, output_dir, CHECK_FLAGS):
    ###########################################
    # Internal Functions
    ###########################################

    ###########################################
    # Main Function
    ###########################################

    # Step 2:
    result_arr = []
    location = shock_location - 1e-3

    while location > flame_location:
        # Extract boundary data at the current location
        boundary_data, _ = pelec_data_extraction(raw_data, location, direction='y', preloaded_grids=None)

        if not boundary_data or 'X' not in boundary_data or 'X Velocity' not in boundary_data:
            print(f"Warning: Missing data at location {location}")
            continue  # Skip if data is incomplete

        vec_length = len(boundary_data['X'])

        if vec_length == 0:
            print(f"Warning: No points found at location {location}")
            continue

        # Determine freestream velocity
        freestream_vel = boundary_data['X Velocity'][vec_length // 2]

        # Compute the velocity threshold (99% of freestream)
        velocity_threshold = 0.99 * freestream_vel

        # Find the first crossing point (from low to high)
        boundary_idx = np.argmax(boundary_data['X Velocity'] >= velocity_threshold)
        """
        boundary_idx = None
        for i in range(1, vec_length):
            if boundary_data['X Velocity'][i] > velocity_threshold:
                boundary_idx = i
                break
        """

        if boundary_idx is None:
            print(f"Warning: No boundary layer threshold crossing found at location {location}")
            continue

        tmp_bl = boundary_data['Y'][boundary_idx]

        # Compute the local Reynolds number
        density = boundary_data['Density'][boundary_idx]
        viscosity = boundary_data['Viscosity'][boundary_idx]

        if viscosity == 0:
            print(f"Warning: Zero viscosity at location {location}, skipping")
            continue

        distance = shock_location - location
        tmp_re_plate = density * np.average(boundary_data['X Velocity']) * distance / viscosity
        tmp_re_pipe = density * np.average(boundary_data['X Velocity']) * 0.000889 / viscosity

        result_arr.append((location, tmp_bl, tmp_re_plate, tmp_re_pipe))

        # Debugging output
        print(f"Location: {location:.4f}, BL Thickness: {tmp_bl:.4f}, Plate Re: {tmp_re_plate:.2e}, Pipe Re: {tmp_re_pipe:.2e}")

        # Update the location
        location -= 1e-3  # Step through locations

    return result_arr

########################################################################################################################
# Function Scripts - Parallel Processed Functions
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
        data, grids = pelec_data_extraction(raw_data, domain_info[1][0][1])

        return time, raw_data, data, grids

    ###########################################
    # Main Function
    ###########################################
    global input_params
    iter_var, const_arr, kwargs = args

    pltFile_dir = iter_var
    domain_info = kwargs.get('domain_info', [])
    output_dir = kwargs.get('output_dir', None)
    CHECK_FLAGS = kwargs.get('CHECK_FLAGS', [])

    # Step 1: Extract the centerline data from the pltfile
    time, raw_data, pre_loaded_data, grids = load_data()

    # Step 2: Determine the flame and lead shock location
    [flame_idx, flame_location] = wave_tracking('Flame',
                                                pre_loaded_data=pre_loaded_data)

    [shock_idx, shock_location] = wave_tracking('Lead Shock',
                                                pre_loaded_data=pre_loaded_data)

    # Step 3:
    boundary_layer_arr = boundary_layer_extractor(raw_data, flame_location, shock_location, output_dir, CHECK_FLAGS)

    return boundary_layer_arr

def pelec_processing(pelec_dirs, domain_info, output_dir, CHECK_FLAGS):
    def write_boundary_layer_data(file_path, data):
        """
        Writes boundary layer data to a file with formatted columns.

        Parameters:
            directory (str): Path to the directory where the file should be saved.
            filename (str): Name of the file to save the data (e.g., 'boundary_layer.txt').
            data (list of tuples): Each tuple should be (location, boundary_layer_thickness, local_Reynolds_number).

        Returns:
            str: Full path of the saved file.
        """

        # Write data to file with formatted output
        with open(file_path, 'w') as f:
            # Header
            f.write(f"{'Location (m)':<25} {'Boundary Layer Thickness (m)':<35} {'Local Reynolds Number':<35}\n")
            f.write("=" * 100 + "\n")

            # Data
            for location, bl_thickness, reynolds in data:
                f.write(f"{location:<25e} {bl_thickness:<35e} {reynolds:<35e}\n")

        return

    ###########################################
    # Main Function
    ###########################################
    # Step 1: Process the individual pelec data files in parallel using multiprocessing
    pelec_data = parallel_processing_function(pelec_dirs, (), single_file_processing,
                                              domain_info=domain_info,
                                              output_dir=output_dir,
                                              CHECK_FLAGS=CHECK_FLAGS)

    # Step 2: Write results to an output file
    write_boundary_layer_data(ensure_long_path_prefix(os.path.join(output_dir, f'Local-Reynolds-Number-Results-V{version}.txt')), pelec_data)

    return


########################################################################################################################
# Main Script
########################################################################################################################

def main():
    # Step 1:
    row_idx = None
    data_set = 'Raw-PeleC'

    CHECK_FLAGS = {
        'Input Variables': ['X', 'Y',
                            'Temperature',
                            'Pressure',
                            'Density',
                            'X Velocity',
                            'Y Velocity']
    }

    # Step 2: Initialize the code with the desired processed variables and mixture composition
    initialize_parameters(
        T=503.15,
        P=10.0 * 100000,
        Phi=1.0,
        Fuel='H2',
        mech='Li-Dryer-H2-mechanism.yaml',
    )
    # Add species if needed (Only use input_params.species if working with a small chemical mechanism file)
    add_species_vars(input_params.species)

    # Step 3: Collect all the present pelec data directories
    dir_path = os.path.dirname(os.path.realpath(__file__))

    time_data_dir = [os.path.join(dir_path, raw_data_folder, time_step)
                     for raw_data_folder in os.listdir(dir_path)
                     if os.path.isdir(os.path.join(dir_path, raw_data_folder)) and raw_data_folder.startswith(f'Raw-{data_set}')
                     for time_step in os.listdir(os.path.join(dir_path, raw_data_folder))
                     if os.path.isdir(os.path.join(dir_path, raw_data_folder, time_step)) and time_step.startswith('plt')]

    dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../')

    time_data_dir = [os.path.join(dir_path, raw_data_folder, time_step)
                     for raw_data_folder in os.listdir(dir_path)
                     if os.path.isdir(os.path.join(dir_path, raw_data_folder)) and raw_data_folder.startswith(
            f'{data_set}')
                     for time_step in os.listdir(os.path.join(dir_path, raw_data_folder))
                     if os.path.isdir(os.path.join(dir_path, raw_data_folder, time_step)) and time_step.startswith(
            'plt')]
    # Step 4: Chronologically order the pltFiles and truncate the raw data list if skip loading is enabled
    #updated_data_list = sort_files(time_data_dir)
    #updated_data_list = [updated_data_list[-1]]
    updated_data_list = time_data_dir

    # Step 5: Determine the domain sizing parameters (size, # of cells)
    domain_info = domain_size_parameters(updated_data_list[-1], row_idx)

    # Step 6: Create the result directories
    os.makedirs(os.path.join(dir_path, f"Processed-Boundary-Layer-Results-V{version}"), exist_ok=True)
    output_dir_path = os.path.join(dir_path, f"Processed-Boundary-Layer-Results-V{version}")

    # Step 8:
    print('Beginning PeleC Processing')
    pelec_processing(updated_data_list, domain_info, output_dir_path, CHECK_FLAGS)
    print('Completed PeleC Processing')

    return

if __name__ == "__main__":
    main()