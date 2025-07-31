import re
import yt
import os
import sys
import time
import numpy as np
import cantera as ct
from mpi4py import MPI
from functools import reduce

from .general import input_params
from .general import *

# Add the parent directory of the current script (your_project) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mpi_work_distrobution.master_slave import Master, Slave
from mpi_work_distrobution.work_queue import WorkQueue

yt.set_log_level(0)
yt.enable_parallelism()

# Optional imports if needed externally:
__all__ = [
    'PELE_VAR_MAP', 'add_species_vars', 'UnitConverter',
    'domain_parameters', 'data_ray_extraction',
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
    for species in input_params.species:
        PELE_VAR_MAP[f'Y({species})'] = {'Name': f'Y({species})', 'Units': ''}
        PELE_VAR_MAP[f'D({species})'] = {'Name': f'D({species})', 'Units': ''}
        PELE_VAR_MAP[f'W({species})'] = {'Name': f'rho_omega_{species}', 'Units': 'g / cm^3 / s'}


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
# Data Import
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
    grid_arr = {
        'x': x_coords,
        'y': y_coords
    }

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
    elif isinstance(desired_y_location, float):
        y_slice_index = np.argmin(abs(y_arr - desired_y_location))
        y_slice_loc = y_arr[y_slice_index]

    else:
        y_slice_index = np.argmin(abs(y_arr - (np.max(y_arr) + np.min(y_arr)) / 2))
        y_slice_loc = y_arr[y_slice_index]

    return (np.array([[0, y_slice_index], [data.ActiveDimensions[0], y_slice_index]]),
            np.array([[data.LeftEdge[0].to_value() / 100, y_slice_loc / 100], [data.RightEdge[0].to_value() / 100, y_slice_loc / 100]]),
            grid_arr)


def cantera_str_acquisition(dr, ray_sort, field_list, idx, gas, missing_str):
    ##########################################
    # Main Function
    ###########################################
    # Step 1: Extract the thermodynamic state and composition
    T = dr["boxlib", "Temp"][ray_sort].to_value().flatten()[idx]
    P = dr["boxlib", "pressure"][ray_sort].to_value().flatten()[idx] / 10

    species_list = [var for var in input_params.species]
    Y = {
        species: dr["boxlib", f"Y({species})"][ray_sort].to_value().flatten()[idx]  # Construct key dynamically
        for species in species_list if f"Y({species})" in np.array(field_list)[:, 1]  # Ensure key exists
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
            missing_data[key] = dr["boxlib", "x_velocity"][ray_sort].to_value().flatten()[idx] / gas.sound_speed
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

def data_ray_extraction(dataset, extract_location, direction='x', flux_calc=False, transport_species=None):

    def ray_processing():
        # Step 1: Extract the ray through the domain using yt
        if direction == 'y':
            dr = dataset.ortho_ray(1, (0, extract_location * 100))
        else:
            dr = dataset.ortho_ray(0, (extract_location * 100, 0))

        # Step 2: Sort the ray based on position
        ray_sort = np.argsort(dr["boxlib", 'x'])

        # Step 3:
        temp_data = {var_info["Name"]: [] for var_info in PELE_VAR_MAP.values()}

        for var in PELE_VAR_MAP:
            var_name = PELE_VAR_MAP[var]["Name"]

            if var in missing_str:
                for i in range(0, len(dr["boxlib", 'x'][ray_sort])):
                    temp_var = cantera_str_acquisition(dr, ray_sort, dataset.field_list, i, gas_missing, missing_str)
                    temp_data[var_name].append(temp_var[var])
            else:
                try:
                    temp_var = dr["boxlib", var_name][ray_sort].to_value().flatten()
                    for vi in temp_var:
                        temp_data[var_name].append(vi)
                except Exception as e:
                    print(f"Missing variable {var_name}: {e}")
                    continue

        # Return result
        return temp_data


    def flux_calculation(data, field_list):

        # Step 1:
        gas = ct.Solution(input_params.mech)

        # Step 3: Calculate the mole fraction spatial gradient
        tmp_mole_frac_grad = {}
        for species in input_params.species:
            tmp_mole_frac_grad[f'X({species})'] = []

        for species in input_params.species:
            tmp_mole_frac_grad[f'X({species})'].append(np.gradient(data[f'X({species})'], data['X']))

        # Step 2:
        flux_dict = {'Mass Flux': [],
                     'Diffusion Flux': [],
                     'Momentum Flux': [],
                     'Energy Flux': []}

        # Step 3:
        for idx in range(len(data['X'])):
            # Step 1:
            T = data['Temperature'][idx]
            P = data['Pressure'][idx]

            species_list = [var for var in input_params.species]
            Y = {
                species: data[f"Y({species})"][idx]  # Construct key dynamically
                for species in species_list if f"Y({species})" in np.array(field_list)[:, 1]  # Ensure key exists
            }

            # Step 2: Modify the gas object
            gas.TPY = T, P, Y

            # Step 1: Save the conservative variables from the conservation equations
            # Mass Flux
            flux_dict['Mass Flux'].append(gas.density_mass * data['X Velocity'][idx])

            # Species Flux
            flux_dict[f'{transport_species} Flux'] = gas.density_mass * gas.Y[gas.species_index(transport_species)]

            # Diffusion Flux (Mixture Averaged)
            j_k_star = np.zeros(len(input_params.species))
            for k, species in enumerate(input_params.species):
                species_idx = gas.species_index(species)
                j_k_star[k] = - gas.density_mass * (gas.molecular_weights[species_idx] / gas.mean_molecular_weight) * \
                              data[f'D({transport_species})'][idx] * tmp_mole_frac_grad[f'X({species})'][idx]

            flux_dict[f'{transport_species} Diffusion Flux'].append(
                j_k_star[input_params.species.index(transport_species)] - data[f'Y({transport_species})'] * np.sum(j_k_star))

            # Momentum FLux
            flux_dict['Momentum Flux'].append(gas.density_mass * (data['X Velocity'][idx] ** 2))

            # Energy Flux
            flux_dict['Energy Flux'].append(gas.density_mass * gas.int_energy_mass)

        return


    ###########################################
    # Main Function
    ###########################################

    global input_params

    # Step 2: Initialize the missing variables
    missing_str = {
        key: raw_var["Name"]
        for key, raw_var in PELE_VAR_MAP.items()
        if raw_var["Name"] not in np.array(dataset.field_list)[:, 1] and key not in {"Grid", "X", "Y"}
    }

    if missing_str:
        gas_missing = ct.Solution(input_params.mech)

    # Step 3: Ensure PELE_VAR_MAP includes the species-specific mappings
    add_species_vars(input_params.species)

    # Step 4:
    tmp_data = ray_processing()

    # Step 9: Merge the data from all levels into a single dataset
    merged_data = {}

    # Iterate over each variable in final_data
    for var, var_info in PELE_VAR_MAP.items():
        var_name = var_info["Name"]

        # Initialize merged x and y arrays with data from max_level
        merged_x = np.array(tmp_data['x'])
        merged_y = np.array(tmp_data[var_name])

        # Now apply the unit conversion for each variable in the merged data
        unit_expr = var_info.get('Units', '')
        if unit_expr:
            converted_y = [UnitConverter.convert(yi, var) for yi in merged_y]
            merged_data[var] = np.array(converted_y)
        else:
            merged_data[var] = merged_y

    # Step 5:
    if flux_calc:
        flux_data = flux_calculation(merged_data)
        merged_data.update(flux_data)  # Add all computed fluxes to merged_data

    return merged_data


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
            try:
                tmp_idx = np.argwhere(data_arr >= 1.01 * data_arr[-1])[-1][0]
            except:
                tmp_idx = 0
            return tmp_idx
        else:
            raise ValueError('Invalid Wave Type! Must be Flame, Maximum Pressure, or Leading Shock')

    ###########################################
    # Main Function
    ###########################################
    # Step 1: Parse input data type

    pre_loaded_data = kwargs.get('pre_loaded_data', None)
    dataset = kwargs.get('dataset', None)

    if pre_loaded_data is None and dataset is None:
        raise ValueError('Either pre_loaded_data or dataset must be provided.')

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

    elif dataset is not None:
        y_loc = kwargs.get('y_loc', None)
        if y_loc is None:
            raise ValueError('y_loc must be provided when raw_data is used.')

        pre_loaded_data = data_ray_extraction(dataset, y_loc)

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