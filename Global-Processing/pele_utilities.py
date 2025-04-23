import re
import yt
import numpy as np
import cantera as ct
from functools import reduce

from general_utilities import input_params
from general_utilities import *

yt.set_log_level(0)

# Optional imports if needed externally:
__all__ = [
    'PELE_VAR_MAP', 'add_species_vars', 'UnitConverter',
    'domain_parameters', 'data_extraction',
    'wave_tracking', 'thermodynamic_state_extractor'
]


#################################################################
# Pele Variable Map
#################################################################

PELE_VAR_MAP = {
    'X': {'Name': 'x', 'Units': 'cm'},
    'Y': {'Name': 'y', 'Units': 'cm'},
    'Temperature': {'Name': 'Temp', 'Units': 'K'},
    'Pressure': {'Name': 'pressure', 'Units': 'g / cm / s^2'},
    'Density': {'Name': 'density', 'Units': 'g / cm^3'},
    'Viscosity': {'Name': 'viscosity', 'Units': 'g / cm / s'},
    'Conductivity': {'Name': 'conductivity', 'Units': 'g cm^2 / s^3 / cm / K'},
    'Sound speed': {'Name': 'soundspeed', 'Units': 'cm / s'},
    'Mach Number': {'Name': 'MachNumber', 'Units': ''},
    'X Velocity': {'Name': 'x_velocity', 'Units': 'cm / s'},
    'Y Velocity': {'Name': 'y_velocity', 'Units': 'cm / s'},
    'Heat Release Rate': {'Name': 'heatRelease', 'Units': 'g cm^2 / s^3 / cm^3'},
    'Cp': {'Name': 'cp', 'Units': 'g cm^2 / s^2 / g / K'},
    'Cv': {'Name': 'cv', 'Units': 'g cm^2 / s^2 / g / K'},
}


def add_species_vars(species_list):
    for species in species_list:
        PELE_VAR_MAP[f'Y({species})'] = ''
        PELE_VAR_MAP[f'D({species})'] = ''
        PELE_VAR_MAP[f'W({species})'] = 'g / cm^3'


#################################################################
# Unit Converter
#################################################################

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
    def convert(cls, value, var_name):
        unit_expr = cls._lookup_unit_expr(var_name)
        num_units, denom_units = cls._parse_units(unit_expr)
        num_factor = cls._convert_units(num_units, 1)
        denom_factor = cls._convert_units(denom_units, -1) if denom_units else 1
        return value * num_factor * denom_factor

    @classmethod
    def _lookup_unit_expr(cls, var_name):
        if var_name not in PELE_VAR_MAP:
            raise ValueError(f"Unknown variable '{var_name}'")
        unit_info = PELE_VAR_MAP[var_name]
        if isinstance(unit_info, dict) and 'Units' in unit_info:
            return unit_info['Units']
        elif isinstance(unit_info, str) and unit_info:
            return unit_info
        raise ValueError(f"Units not defined for variable '{var_name}'.")

    @classmethod
    def _parse_units(cls, unit_expr):
        terms = re.split(r'\s*/\s*', unit_expr)
        num_units = terms[0].split()
        denom_units = [] if len(terms) == 1 else [t for term in terms[1:] for t in term.split()]
        return num_units, denom_units

    @classmethod
    def _convert_units(cls, units, exponent):
        if not units:
            return 1
        factors = []
        for unit in units:
            match = re.match(r"([a-zA-Z]+)(?:\^(-?\d+))?", unit)
            if not match:
                raise ValueError(f"Invalid unit format: {unit}")
            base_unit, exp = match.groups()
            if base_unit not in cls.UNIT_MAP:
                raise ValueError(f"Unknown unit '{base_unit}'")
            base_factor = cls.UNIT_MAP[base_unit]
            power = int(exp) if exp else 1
            factors.append(base_factor ** (power * exponent))
        return reduce(lambda x, y: x * y, factors, 1)


#################################################################
# Data Extraction
#################################################################

def domain_parameters(directory_path, desired_y_location=None):
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


def cantera_str_acquisition(raw_data, grid, idx, gas, missing_str):
    ##########################################
    # Main Function
    ###########################################
    # Step 1: Extract the thermodynamic state and composition
    T = grid["boxlib", "Temp"].to_value().flatten()[idx]
    P = grid["boxlib", "pressure"].to_value().flatten()[idx] / 10

    species_list = [var for var in input_params.species]
    Y = {
        species: grid["boxlib", f"Y({species})"].to_value().flatten()[idx]  # Construct key dynamically
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


def parallel_level_data_extraction(args):
    ##########################################
    # Main Function
    ###########################################
    # Step 1: Unpack the arguments
    iter_var, const_arr, kwargs = args

    level = iter_var
    data_dir, direction, extract_location, missing_str = const_arr
    input_params = kwargs["input_params"]
    PELE_VAR_MAP = kwargs["PELE_VAR_MAP"]
    preloaded_grids = kwargs.get('preloaded_grids', None)

    # Step 2: Load the data and any preloaded grids
    raw_data = yt.load(data_dir)
    used_grids = {} if preloaded_grids is None else preloaded_grids

    # Get the maximum refinement level and smallest grid spacing
    max_level = raw_data.index.max_level
    dx_min = raw_data.index.get_smallest_dx()
    dx = dx_min.to_value() * 2 ** (max_level - level)  # Grid spacing at this level

    # Step 3: Create a mapping for the variables
    level_data = {level: {PELE_VAR_MAP[var]["Name"]: {} for var in PELE_VAR_MAP.keys()}}

    # Step 4: Check if missing_str is not empty and create cantera object
    if missing_str:
        gas_missing = ct.Solution(input_params.mech)

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
                    temp_var = cantera_str_acquisition(grid, i, gas_missing, missing_str)
                    level_data[level][var_name][xi] = temp_var[var]  # Higher levels overwrite lower ones
            else:
                try:
                    temp_var = grid["boxlib", var_name].flatten()  # Use the mapped name for lookup
                    for xi, vi in zip(position_arr[mask], temp_var[mask]):
                        level_data[level][var_name][xi] = vi  # Store using mapped name
                except:
                    continue

    return (level_data, used_grids)


def data_extraction(data_dir, extract_location, direction='x', raw_data=None, preloaded_grids=None, parallelize=False):
    ###########################################
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
    global input_params

    # Step 1: Load the data if not provided
    raw_data = yt.load(data_dir) if raw_data is None else raw_data

    # Step 2: Create mapping for the variables
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

    # Get the maximum refinement level and smallest grid spacing
    max_level = raw_data.index.max_level
    dx_min = raw_data.index.get_smallest_dx()
    # Dictionary to store refined data per level
    # Create level_data using the 'Name' field from PELE_VAR_MAP
    used_grids = {} if preloaded_grids is None else preloaded_grids
    # If grids are preloaded, process only those levels
    levels_to_process = reversed(preloaded_grids.keys()) if preloaded_grids else range(max_level, -1, -1)


    if parallelize:
        tmp_results = parallel_processing_function(levels_to_process,
                                               (data_dir, direction, extract_location, missing_str),
                                               parallel_level_data_extraction, n_procs=max_level,
                                               input_params=input_params,
                                               PELE_VAR_MAP=PELE_VAR_MAP,
                                               preloaded_grids=preloaded_grids)
    else:
        tmp_results = parallel_processing_function(levels_to_process,
                                                  (data_dir, direction, extract_location, missing_str),
                                                  parallel_level_data_extraction, n_procs=1,
                                                  input_params=input_params,
                                                  PELE_VAR_MAP=PELE_VAR_MAP,
                                                  preloaded_grids=preloaded_grids)

    # Merge the data from all levels into a single dataset
    used_grids = {}
    level_data = {}
    for tmp_lvl_data, tmp_used_grids in tmp_results:
        level_data.update(tmp_lvl_data)
        used_grids.update(tmp_used_grids)

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


#################################################################
# Wavefront Extraction
#################################################################

def wave_tracking(wave_type, **kwargs):
    ###########################################
    # Internal Functions
    ###########################################
    def find_wave_index(wave_type):
        if wave_type == 'Flame':
            tmp_arr = np.zeros(2, dtype=int)
            try:
                tmp_arr[0] = np.argwhere(data_arr[0, :] >= flame_temp)[-1]
                tmp_arr[1] = np.argmax(data_arr[1, :])

                if abs(tmp_arr[0] - tmp_arr[1]) > 10:
                    print('Warning: Flame Location differs by more than 10 cells!\n'
                            'Flame Temperature Location', x_arr[tmp_arr[0]],
                            '\nFlame Species Location', x_arr[tmp_arr[1]])
                return tmp_arr[1]
            except:
                print('Error: Could not find Y_H to determine Flame Location!')
                tmp_arr[0] = np.argwhere(data_arr[0, :] >= flame_temp)[-1]
                return tmp_arr[0]
        elif wave_type == 'Shock':
            return np.argwhere(data_arr >= 1.01 * data_arr[-1])[-1][0]
        else:
            raise ValueError('Invalid Wave Type! Must be Flame, Maximum Pressure, or Leading Shock')

    ###########################################
    # Main Function
    ###########################################
    # Step 1: Parse input data type

    pre_loaded_data = kwargs.get('pre_loaded_data', None)
    data_dir_path = kwargs.get('data_dir_path', None)

    if pre_loaded_data is None and data_dir_path is None:
        raise ValueError('Either pre_loaded_data or data_dir_path must be provided.')

    # Step 2: Extract flame temperature from kwargs or use default
    flame_temp = kwargs.get('flame_temp', 2000)

    # Step 3: Extract preloaded data if available, otherwise load data
    if pre_loaded_data is not None:
        x_arr = pre_loaded_data['X']
        # Use the preloaded data directly
        if wave_type == 'Flame':
            data_arr = np.empty((2, len(x_arr)), dtype=np.float64)
            data_arr[0, :] = pre_loaded_data['Temperature']
            data_arr[1, :] = pre_loaded_data['Y(HO2)']
        else:
            data_arr = pre_loaded_data['Pressure']

    elif data_dir_path is not None:
        y_loc = kwargs.get('y_loc', None)
        if y_loc is None:
            raise ValueError('y_loc must be provided when raw_data is used.')

        pre_loaded_data = data_extraction(data_dir_path, y_loc)

        x_arr = pre_loaded_data['X']
        # Use the preloaded data directly
        if wave_type == 'Flame':
            data_arr = np.empty((2, len(x_arr)), dtype=np.float64)
            data_arr[0, :] = pre_loaded_data['Temperature']
            data_arr[1, :] = pre_loaded_data['Y(HO2)']
        else:
            data_arr = pre_loaded_data['Pressure']

    else:
        raise ValueError('No valid data provided for wave tracking.')

    # Step 4: Find the index of the wave
    wave_idx = find_wave_index(wave_type)

    if pre_loaded_data is not None:
        return wave_idx, x_arr[wave_idx]
    else:
        return wave_idx, x_arr[wave_idx] / 100, y_idx


def thermodynamic_state_extractor(pre_loaded_data, wave_loc, offset):
    ###########################################
    # Main Function
    ###########################################
    # Step 1: Determine the location of the wave
    try:
        probe_idx = np.argmin(abs(pre_loaded_data['X'] - (wave_loc + offset)))
    except Exception as e:
        print(f"Error Could not find thermodynamic location: {e}")
        probe_idx = -1

    # Step 2: Return the thermodynamic state
    tmp_dict = {}
    tmp_dict['Temperature'] = pre_loaded_data['Temperature'][probe_idx]
    tmp_dict['Pressure'] = pre_loaded_data['Pressure'][probe_idx]
    tmp_dict['Density'] = pre_loaded_data['Density'][probe_idx]
    tmp_dict['Sound Speed'] = pre_loaded_data['Sound speed'][probe_idx]

    return tmp_dict
