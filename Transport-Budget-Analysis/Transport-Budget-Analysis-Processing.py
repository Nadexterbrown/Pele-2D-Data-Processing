import re
import os
import yt
import time
import numpy as np
import cantera as ct
from mpi4py import MPI
from typing import Optional
from datetime import datetime
from collections import defaultdict
from scipy.interpolate import CubicSpline
from dataclasses import dataclass, field, fields
from yt.utilities.parallel_tools.parallel_analysis_interface import parallel_objects, communication_system


from sdtoolbox.znd import zndsolve
from sdtoolbox.utilities import CJspeed_plot
from sdtoolbox.postshock import CJspeed, PostShock_fr, PostShock_eq

from utilities.general import *
from utilities.pele import *

yt.set_log_level(0)

#################################################################
# MPI Global Parameters
#################################################################

# Internal MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
print(f"Rank {rank} of {size} is running.", flush=True)

########################################################################################################################
# Script Configuration Classes
########################################################################################################################

@dataclass
class DataExtractionConfig:
    DataType: Optional[str] = None
    Flag: Optional[bool] = False
    Stride: Optional[int] = 1
    _DataPath: Optional[str] = field(default=None, init=False, repr=False)

    @property
    def DataPath(self) -> Optional[str]:
        return self._DataPath

    @DataPath.setter
    def DataPath(self, path: Optional[str]):
        if path is None:
            self._DataPath = None
        elif os.path.isabs(path):
            self._DataPath = path
        else:
            base_path = os.path.dirname(os.path.realpath(__file__))
            self._DataPath = os.path.normpath(os.path.join(base_path, path))

@dataclass
class ScriptConfig:
    FreeFlame: DataExtractionConfig = field(default_factory=lambda: DataExtractionConfig(DataType='Cantera Flame'))
    Detonation: DataExtractionConfig = field(default_factory=lambda: DataExtractionConfig(DataType='SDToolbox Detonation'))
    SLTORC: DataExtractionConfig = field(default_factory=lambda: DataExtractionConfig(DataType='SLTORC'))
    Pele: DataExtractionConfig = field(default_factory=lambda: DataExtractionConfig(DataType='Pele'))

########################################################################################################################
# Cantera Flame/Detonation Simulation
########################################################################################################################

def cantera_flame(transport_species):

    ###########################################
    # Internal Functions
    ###########################################

    def dict_creation():
        # Step 1: Save the primitive variables from the conservation equations and the relevant thermodynamic parameters
        data = {
            'X': np.array([x for x in f.grid]),
            'X Velocity': f.velocity,
            'Temperature': f.T,
            'Pressure': f.P,
            'Density': f.density_mass,
            f'D({transport_species})': f.mix_diff_coeffs_mass[gas.species_index(transport_species)],
            f'h({transport_species})': f.standard_enthalpies_RT[gas.species_index(transport_species)] * ct.gas_constant * f.T,
            f'W({transport_species})': f.net_production_rates[gas.species_index(transport_species)] * gas.molecular_weights[gas.species_index(transport_species)]
        }

        for species in input_params.species:
            data[f'Y({species})'] = f.Y[gas.species_index(species)]
            data[f'X({species})'] = f.X[gas.species_index(species)]

        return data

    ###########################################
    # Main Function
    ###########################################

    # Step 1: Simulation parameters
    width = 0.001  # m
    loglevel = 0  # amount of diagnostic output (0 to 8)
    # Step 2: Solution object used to compute mixture properties, set to the state of the
    #         upstream fuel-air mixture
    gas = ct.Solution(input_params.mech)
    gas.TPX = input_params.T, input_params.P, input_params.X
    # Step 3: Set up flame object
    f = ct.FreeFlame(gas, width=width)
    f.set_refine_criteria(ratio=3, slope=0.01, curve=0.01)
    # Step 4: Solve with mixture-averaged transport model
    f.transport_model = 'mixture-averaged'
    f.solve(loglevel=loglevel, auto=True)

    return dict_creation()


def sdtoolbox_detonation(transport_species):
    ###########################################
    # Internal Functions
    ###########################################

    def dict_creation(transport_species):
        # Initialize an empty dictionary with NumPy arrays
        data = {
            'X': np.zeros(len(out['distance'])),
            'X Velocity': np.zeros(len(out['distance'])),
            'Temperature': np.zeros(len(out['distance'])),
            'Pressure': np.zeros(len(out['distance'])),
            'Density': np.zeros(len(out['distance'])),
            f'D({transport_species})': np.zeros(len(out['distance'])),
            f'h({transport_species})': np.zeros(len(out['distance'])),
            f'W({transport_species})': np.zeros(len(out['distance']))
        }

        # Add species-dependent properties dynamically
        for species in input_params.species:
            data[f'Y({species})'] = np.zeros(len(out['distance']))
            data[f'X({species})'] = np.zeros(len(out['distance']))

        # Create a gas object
        gas_tmp = ct.Solution(input_params.mech)

        for i in range(len(out['distance'])):
            # Set gas state for the current step
            species_dict = {input_params.species[j]: out['species'][j][i] for j in range(len(input_params.species))}
            gas_tmp.TPY = out['T'][i], out['P'][i], species_dict

            # Store computed values directly into the NumPy arrays
            data['X'][i] = out['distance'][i]
            data['X Velocity'][i] = out['U'][i]
            data['Temperature'][i] = out['T'][i]
            data['Pressure'][i] = out['P'][i]
            data['Density'][i] = gas_tmp.density_mass
            data[f'D({transport_species})'][i] = gas_tmp.mix_diff_coeffs_mass[gas_tmp.species_index(transport_species)]  # Diffusion Coefficient
            data[f'h({transport_species})'][i] = gas_tmp.standard_enthalpies_RT[gas_tmp.species_index(transport_species)] * ct.gas_constant * gas_tmp.T  # Enthalpy
            data[f'W({transport_species})'][i] = gas_tmp.net_production_rates[gas_tmp.species_index(transport_species)] * gas_tmp.molecular_weights[gas_tmp.species_index(transport_species)]  # Reaction Rate

            for species in input_params.species:
                data[f'Y({species})'][i] = gas_tmp.Y[gas_tmp.species_index(species)]  # Mass Fraction
                data[f'X({species})'][i] = gas_tmp.X[gas_tmp.species_index(species)]  # Mass Fraction

        return data


    ###########################################
    # Main Function
    ###########################################

    # Find CJ speed and related data, make CJ diagnostic plots
    cj_speed, R2, plot_data = CJspeed(input_params.P, input_params.T, input_params.X, input_params.mech, fullOutput=True)
    CJspeed_plot(plot_data, cj_speed)

    # Set up gas object
    gas1 = ct.Solution(input_params.mech)
    gas1.TPX = input_params.T, input_params.P, input_params.X

    # Find equilibrium post shock state for given speed
    gas = PostShock_eq(cj_speed, input_params.P, input_params.T, input_params.X, input_params.mech)
    u_cj = cj_speed * gas1.density / gas.density

    # Find frozen post shock state for given speed
    gas = PostShock_fr(cj_speed, input_params.P, input_params.T, input_params.X, input_params.mech)

    # Solve ZND ODEs, make ZND plots
    out = zndsolve(gas, gas1, cj_speed, t_end=1e-6, advanced_output=True)

    return dict_creation()


########################################################################################################################
# 1D SLTORC Data Extraction
########################################################################################################################

def sltorc_data_loading(dir_path, transport_species, stride=1):
    ###############################################################################
    # Internal Functions
    ###############################################################################

    def parse_data_file(filepath):
        with open(filepath, 'r') as f:
            lines = f.read().splitlines()

        # Extract column headers (line starting with '#' and 'psi')
        header_line = next((l for l in lines if l.strip().startswith('#') and 'psi' in l), None)
        if not header_line:
            raise ValueError("Header line with variable names not found.")
        headers = re.split(r'\s{2,}|\t+', header_line.lstrip('#').strip())

        # Split file into blocks per time step (separated by 2+ blank lines)
        content = "\n".join(lines)
        blocks = re.split(r'(?:\r?\n){2,}', content.strip())

        result = defaultdict(list)

        for block in blocks:
            lines = block.strip().splitlines()
            time_line = next((l for l in lines if re.match(r"#\d+\s*-\s*t=.*ms", l.strip())), None)
            if not time_line:
                continue

            match = re.match(r"#\d+\s*-\s*t=([\d.+-eE]+)ms", time_line.strip())
            if not match:
                raise ValueError(f"Could not extract time from line: {time_line}")

            time_value = float(match.group(1))
            time_step_values = []

            for line in lines[lines.index(time_line) + 1:]:
                if line.strip().startswith('#') or not line.strip():
                    continue
                values = re.split(r'\s+', line.strip())
                if len(values) != len(headers):
                    raise ValueError(f"Data/header mismatch at time={time_value}: {values}")
                time_step_values.append([float(v) for v in values])

            for header, values in zip(headers, zip(*time_step_values)):
                result[header].append(list(values))

            result["time"].append(time_value)

        return dict(result)

    def cantera_state(X_list, T_list, D_list, Y_list, input_params):

        # Step 1:
        gas_tmp = ct.Solution(input_params.mech)

        # Step 2:
        result = {
            'X': [],
            'X Velocity': [],
            'Temperature': [],
            'Pressure': [],
            'Density': [],
            f'D({transport_species})': [],
            f'h({transport_species})': [],
            f'W({transport_species})': []

        }

        for species in input_params.species:
            result[f'Y({species})'] = []
            result[f'X({species})'] = []

        for idx in range(len(T_list)):
            T = T_list[idx]
            D = D_list[idx]
            Y = {species: Y_list[species][idx] for species in input_params.species}

            gas_tmp.TDY = T, D, Y

            result['X'].append(X_list[idx])
            result['X Velocity'].append(0.0)
            result['Temperature'].append(gas_tmp.T)
            result['Pressure'].append(gas_tmp.P)
            result['Density'].append(gas_tmp.density_mass)
            result[f'D({transport_species})'].append(gas_tmp.mix_diff_coeffs_mass[gas_tmp.species_index(transport_species)])
            result[f'h({transport_species})'].append(gas_tmp.standard_enthalpies_RT[gas_tmp.species_index(transport_species)] * ct.gas_constant * gas_tmp.T)
            result[f'W({transport_species})'].append(gas_tmp.net_production_rates[gas_tmp.species_index(transport_species)] * gas_tmp.molecular_weights[gas_tmp.species_index(transport_species)])

            for species in input_params.species:
                result[f'Y({species})'].append(gas_tmp.Y[gas_tmp.species_index(species)])
                result[f'X({species})'].append(gas_tmp.X[gas_tmp.species_index(species)])

        for key in result:
            result[key] = np.array(result[key])

        return result

    ###############################################################################
    # Main Function
    ###############################################################################

    if rank == 0:
        raw_data = parse_data_file(os.path.join(dir_path, 'output-alt.dat'))
    else:
        raw_data = None

    raw_data = comm.bcast(raw_data, root=0)
    comm.Barrier()

    time_indices = list(range(0, len(raw_data['time']), stride))

    local_results = {}
    for sto, i in parallel_objects(time_indices, njobs=size, storage=local_results):
        T = raw_data['Temperature, K'][i]
        D = raw_data['Density, kg/m3'][i]
        Y = {species: raw_data[species][i] for species in input_params.species}
        X = raw_data['r, m'][i]

        sto.result_id = int(i)
        sto.result = {
            'Time': raw_data['time'][i],
            'Data': cantera_state(X, T, D, Y, input_params)
        }

    return local_results


########################################################################################################################
# Data Extraction Parent Function
########################################################################################################################

def data_extraction(transport_species, script_config):

    # Step 1: Required variables
    base_vars = ['X', 'Temperature', 'Pressure', 'X Velocity', f'W({transport_species})']
    species_vars = [f'Y({s})' for s in input_params.species]
    required_vars = base_vars + species_vars

    # Step 2: Determine which configuration is active
    data_dict = {}

    if script_config.FreeFlame.Flag and rank == 0:
        # Step 2.1: Extract and filter data
        tmp_data = cantera_flame(transport_species)
        tmp_data_dict = {var: tmp_data[var] for var in required_vars if var in tmp_data}
        # Step 2.2: Populate time-labeled entries
        for t in range(4):
            data_dict[t] = {
                'Time': float(t),
                'Data': tmp_data_dict
            }

    elif script_config.Detonation.Flag and rank == 0:
        # W.I.P.
        print('Detonation case: Work in Progress')

    elif script_config.SLTORC.Flag:
        if rank == 0:
            rank_log(f"Beginning SLTORC Data Extraction")
        # Step 2.1: Extract and filter SLTORC data
        tmp_data = sltorc_data_loading(
            script_config.SLTORC.DataPath,
            transport_species,
            stride=script_config.SLTORC.Stride)
        data_dict = {var: tmp_data[var] for var in required_vars if var in tmp_data}

    elif script_config.Pele.Flag:
        if rank == 0:
            rank_log(f"Beginning Pele Data Extraction")
        # Step 2.1: Determine file paths for each time step
        paths = load_directories(script_config.Pele.DataPath)
        sorted_paths = sort_files(paths)[::script_config.Pele.Stride]
        pltname_list = [os.path.basename(p) for p in sorted_paths]
        # Step 3.1: Extract data
        for sto, dir in yt.parallel_objects(sorted_paths, -1, storage=data_dict):
            # Step 4.1: Load the data
            ds = yt.load(dir)
            # Step 4.2: Extract and filter Pele data
            tmp_data = data_ray_extraction(ds, 0.005 / 100)
            tmp_data_dict = {var: tmp_data[var] for var in required_vars if var in tmp_data}
            rank_log(f"{ds.basename} Processed")
            # Step 4.4: Export Data
            sto.result_id = pltname_list.index(ds.basename)
            sto.result = {
                'Time': ds.current_time.to_value(),
                'Data': tmp_data_dict
            }
    else:
        if rank == 0:
            print('No data extraction flag was set in script_config.')

    # If the case was serial-only, bcast the data_dict from rank 0 to everyone
    if not script_config.Pele.Flag or script_config.SLTORC.Flag:
        data_dict = None
        data_dict = comm.bcast(data_dict, root=0)

    if rank == 0:
        rank_log(f"Data Extraction Complete")
    return data_dict


########################################################################################################################
# Transport Budget Analysis (CDR)
########################################################################################################################

def transport_budget_analysis(data, transport_species, output_path, script_config):

    ###############################################################################
    # Internal Functions
    ###############################################################################

    def flux_calculation():
        # Step 1: Distribute data across all ranks
        local_data = comm.bcast(data, root=0)

        # Step 2:
        tmp_results = {}
        for sto, i in parallel_objects(local_data.keys(), njobs=size, storage=tmp_results):
            # Define the local cantera gas object
            gas_tmp = ct.Solution(input_params.mech)

            T_tmp = local_data[i]['Data']['Temperature']
            P_tmp = local_data[i]['Data']['Pressure']
            Y_tmp = {f'Y({species})': local_data[i]['Data'][f'Y({species})']
                     for species in input_params.species}
            V_tmp = local_data[i]['Data']['X Velocity']

            X_tmp = {}
            for species in input_params.species:
                X_tmp[f'X({species})'] = []

            for idx in range(len(local_data[i]['Data']['X'])):
                Y_vec =  {f'{species}': Y_tmp[f'Y({species})'][idx] for species in input_params.species}

                gas_tmp.TPY = T_tmp[idx], P_tmp[idx], Y_vec
                for species in input_params.species:
                    X_tmp[f'X({species})'].append(gas_tmp.X[gas_tmp.species_index(species)])

            # Mole Fraction Gradient
            tmp_mole_frac_grad = {f'X({species})': [] for species in input_params.species}
            for k, species in enumerate(input_params.species):
                tmp_mole_frac_grad[f'X({species})'] = np.gradient(X_tmp[f'X({species})'], local_data[i]['Data']['X'])

            # Flux Calculation
            tmp_mass_flux, tmp_species_flux, tmp_diffusion_flux, tmp_momentum_flux, tmp_energy_flux = [], [], [], [], []
            for idx in range(len(local_data[i]['Data']['X'])):
                # Step gas object
                gas_tmp.TPY = T_tmp[idx], P_tmp[idx], {species: Y_tmp[f"Y({species})"][idx] for species in input_params.species}
                transport_species_cantera_idx = gas_tmp.species_index(transport_species)

                # Mass Flux
                tmp_mass_flux.append(gas_tmp.density_mass * V_tmp[idx])
                # Species (Transport) Flux
                tmp_species_flux.append(gas_tmp.density_mass * gas_tmp.Y[transport_species_cantera_idx])
                # Diffusion Flux (Mixture Averaged)
                transport_species_input_idx = input_params.species.index(transport_species)
                j_k_star = np.zeros(len(input_params.species))
                for k, species in enumerate(input_params.species):
                    species_idx = gas_tmp.species_index(species)
                    j_k_star[k] = - gas_tmp.density_mass * (
                                gas_tmp.molecular_weights[species_idx] / gas_tmp.mean_molecular_weight) * \
                                  gas_tmp.mix_diff_coeffs_mass[species_idx] * tmp_mole_frac_grad[f'X({species})'][idx]

                tmp_diffusion_flux.append(j_k_star[transport_species_input_idx] - gas_tmp.Y[transport_species_cantera_idx] * np.sum(j_k_star))
                # Momentum FLux
                tmp_momentum_flux.append(gas_tmp.density_mass * (V_tmp[idx] ** 2))
                # Energy Flux
                tmp_energy_flux.append(gas_tmp.density_mass * gas_tmp.int_energy_mass)

            sto.result_id = int(i)
            sto.result = {
                'Time': local_data[i]['Time'],
                'Mass Flux': tmp_mass_flux,
                'Species Flux': tmp_species_flux,
                'Diffusion Flux': tmp_diffusion_flux,
                'Momentum FLux': tmp_momentum_flux,
                'Energy Flux': tmp_energy_flux
            }

        return tmp_results


    def unsteady_term():

        ###############################################################################
        # Internal Functions
        ###############################################################################

        def data_interpolation(x, y, x_interp):
            # Create cubic spline with extrapolation disabled
            spline = CubicSpline(x, y, extrapolate=False)

            # Interpolate values (in-range get interpolated, out-of-range become nan)
            y_interp = spline(x_interp)

            # Manually clamp out-of-bounds: fill with first/last y value
            y_interp = np.where(x_interp < x[0], y[0], y_interp)
            y_interp = np.where(x_interp > x[-1], y[-1], y_interp)

            return y_interp


        def pele_griding(time_arr, pos_arr, flux_arr, idx):

            #####################################
            # Internal Functions
            #####################################

            def find_refined_boundaries(grid, dx_level0=None, spike_threshold=1e-12):
                # First derivative (spacing between grid points)
                dx = np.gradient(grid)
                # Second derivative (change in spacing — detects refinement interfaces)
                ddx = np.gradient(dx)
                # Identify where second derivative spikes
                spike_indices = np.where(np.abs(ddx) > spike_threshold)[0]
                if spike_indices.size == 0:
                    return None, None
                # Define the boundaries
                left_idx = spike_indices[0]
                right_idx = spike_indices[-1]
                left_boundary = grid[left_idx]
                right_boundary = grid[right_idx] if right_idx < len(grid) else grid[-1]
                return left_boundary, right_boundary


            #####################################
            # Main Function
            #####################################

            tmp_grids = {0: {'Grid': pos_arr[0],
                             'dx': np.diff(pos_arr[0])},
                         1: {'Grid': pos_arr[1],
                             'dx': np.diff(pos_arr[1])},
                         2: {'Grid': pos_arr[2],
                             'dx': np.diff(pos_arr[2])}}

            boundaries = {}
            for key in tmp_grids:
                left, right = find_refined_boundaries(tmp_grids[key]['Grid'], dx_level0=np.max(tmp_grids[1]['dx']))
                boundaries[key] = {'left': left, 'right': right}

            # Determine global min/max boundaries
            left_bound = np.min([g['Grid'][0] for g in tmp_grids.values()])
            right_bound = np.max([g['Grid'][-1] for g in tmp_grids.values()])
            refined_left = np.min([b['left'] for b in boundaries.values() if b['left'] is not None])
            refined_right = np.max([b['right'] for b in boundaries.values() if b['right'] is not None])

            # Build the grid in three regions
            grid_left = np.arange(left_bound, refined_left, np.max(tmp_grids[1]['dx']))
            grid_mid = np.arange(refined_left, refined_right, np.min(tmp_grids[1]['dx']))
            grid_right = np.arange(refined_right, right_bound + np.max(tmp_grids[1]['dx']), np.max(tmp_grids[1]['dx']))

            # Remove duplicates at junctions
            if grid_left.size > 0 and grid_mid.size > 0 and np.isclose(grid_left[-1], grid_mid[0]):
                grid_mid = grid_mid[1:]
            if grid_mid.size > 0 and grid_right.size > 0 and np.isclose(grid_mid[-1], grid_right[0]):
                grid_right = grid_right[1:]

            hybrid_grid = np.concatenate([grid_left, grid_mid, grid_right])

            # Interpolate the flux data onto the uniform grid for each time step
            tmp_flux_data = np.array([
                data_interpolation(pos_arr[0], flux_arr[0], hybrid_grid),
                data_interpolation(pos_arr[1], flux_arr[1], hybrid_grid),
                data_interpolation(pos_arr[2], flux_arr[2], hybrid_grid)
            ])  # Shape: (3, N)

            # Compute time derivative
            dfdt = np.gradient(tmp_flux_data, time_arr, axis=0)[idx]

            return data_interpolation(hybrid_grid, dfdt, pos_arr[idx])


        def general_griding(time_arr, pos_arr, flux_arr, idx):

            #####################################
            # Internal Functions
            #####################################

            def grid_regularization(grid_data):
                """
                    Determine the smallest grid spacing across all spatial grids from multiple ranks.

                    Parameters:
                    -----------
                    data : list of lists
                        A list of lists, where each list contains a 1D array representing spatial grids.

                    Returns:
                    --------
                    dx_min : float
                        The smallest spacing found across all x arrays.
                    x_uniform : 1D array
                        A common uniform grid spanning the full domain using dx_min.
                """

                dx_min = []
                x_min = []
                x_max = []

                # Iterate over all lists of spatial grids
                for x_arr in grid_data:
                    x_min.append(np.min(x_arr))
                    x_max.append(np.max(x_arr))
                    dx_min.append(np.min(np.diff(x_arr)))

                # Find the overall minimum dx and create a uniform grid using the minimum dx
                min_dx = np.min(dx_min)
                uniform_grid_start = np.min(x_min)
                uniform_grid_end = np.max(x_max)

                # Create the uniform grid with the smallest spacing
                x_uniform = np.arange(uniform_grid_start, uniform_grid_end, min_dx)

                return x_uniform


            #####################################
            # Main Function
            #####################################

            # Create the uniform grid for the current time (i)
            x_uniform = grid_regularization(pos_arr)

            # Interpolate the flux data onto the uniform grid for each time step
            tmp_flux_data = np.array([
                data_interpolation(pos_arr[0], flux_arr[0], x_uniform),
                data_interpolation(pos_arr[1], flux_arr[1], x_uniform),
                data_interpolation(pos_arr[2], flux_arr[2], x_uniform)
            ])  # Shape: (3, N)

            # Compute time derivative
            dfdt = np.gradient(tmp_flux_data, time_arr, axis=0)[idx]

            return data_interpolation(x_uniform, dfdt, pos_arr[idx])


        ###############################################################################
        # Main Function
        ###############################################################################

        # Step 1: Distribute data across all ranks
        local_data = comm.bcast(data, root=0)

        tmp_results = {}

        for sto, i in parallel_objects(local_data.keys(), njobs=size, storage=tmp_results):
            # Step 1: Extract the adjacent time step grids
            if i > 0 and i < len(local_data.keys()) - 1:
                tmp_time = [local_data[j]['Time'] for j in [i - 1, i, i + 1]]
                tmp_pos = [local_data[j]['Data']['X'] for j in [i - 1, i, i + 1]]
                tmp_flux = [flux_data[j]['Species Flux'] for j in [i - 1, i, i + 1]]
                tmp_idx = 1

            elif i == 0:
                tmp_time = [local_data[j]['Time'] for j in [i, i + 1, i + 2]]
                tmp_pos = [local_data[j]['Data']['X'] for j in [i, i + 1, i + 2]]
                tmp_flux = [flux_data[j]['Species Flux'] for j in [i, i + 1, i + 2]]
                tmp_idx = 0

            elif i == len(local_data.keys()) - 1:
                tmp_time = [local_data[j]['Time'] for j in [i - 2, i - 1, i]]
                tmp_pos = [local_data[j]['Data']['X'] for j in [i - 2, i - 1, i]]
                tmp_flux = [flux_data[j]['Species Flux'] for j in [i - 2, i - 1, i]]
                tmp_idx = 2

            else:
                print('Error')

            if data_type == 'Pele':
                dfdt = pele_griding(tmp_time, tmp_pos, tmp_flux, tmp_idx)
            else:
                dfdt = general_griding(tmp_time, tmp_pos, tmp_flux, tmp_idx)

            # Store result
            sto.result_id = int(i)
            sto.result = {
                'Time': tmp_time[tmp_idx],
                'dfdt': dfdt
            }

        comm.Barrier()
        return tmp_results


    ###############################################################################
    # Main Function
    ###############################################################################

    if rank == 0:
        rank_log(f"Beginning Temporal and Flux Calculations")

    data_type = next(
        (name for name, config in vars(script_config).items() if getattr(config, 'Flag', False)),
        None
    )

    # Determine all conservation fluxes
    flux_data = flux_calculation()

    # Step 2:
    drhoYdt = unsteady_term()

    # Step 2:
    if rank == 0:
        print('Data Keys:', data.keys(), flush=True)
        print('dfdt Keys:', drhoYdt.keys(), flush=True)
        for i, key in enumerate(data.keys()):
            # Step 2.1:
            tmp_data = data[key]['Data']
            tmp_flux_data = flux_data[key]
            # Step 2.3:
            C = np.gradient(tmp_flux_data['Mass Flux'] * tmp_data[f'Y({transport_species})'], tmp_data['X'])
            # Step 2.4:
            D = np.gradient(tmp_flux_data['Diffusion Flux'], tmp_data['X'])
            # Step 2.5:
            R = tmp_data[f'W({transport_species})']
            # Step 2.6:
            if data_type == 'Pele':
                total = (drhoYdt[key]['dfdt'] + C + D + R)
            else:
                total = (drhoYdt[key]['dfdt'] + C + D - R)

            #
            plot_center = tmp_data['X'][np.argmax(tmp_data['Y(HO2)'])]
            animation_frame_generation(tmp_data['X'], (drhoYdt[key]['dfdt'], C, D, R, total),
                                       ('drhoY/dt', 'C', 'D', 'R', 'Total'),
                                       os.path.join(output_path, f'Overall-Plot-{key}.png'),
                                       split_axis=False, plot_center=plot_center, window_size=5e-5)
            animation_frame_generation(tmp_data['X'], tmp_data['Temperature'],
                                       'Temperature',
                                       os.path.join(output_path, f'Overall-Temperature-Plot-{key}.png'),
                                       split_axis=False)

    return


########################################################################################################################
# Main Script
########################################################################################################################

def main():
    # Step 1: Set the processing parameters
    script_config = ScriptConfig()
    #
    script_config.Pele.Flag = True
    script_config.Pele.DataPath = '../2D-Pele-Test-Data'
    script_config.Pele.Stride = 2

    # Step 2: Define the input parameters
    transport_species = 'HO2'
    initialize_parameters(
        T=503.15,
        P=10.0 * 100000,
        Phi=1.0,
        Fuel='H2',
        mech='../Chemical-Mechanisms/Li-Dryer-H2-mechanism.yaml',
    )  # ✅ shared state
    """
    initialize_parameters(
        T=300,
        P=1 * ct.one_atm,
        Phi=1.0,
        Fuel='C2H6',
        mech='../Chemical-Mechanisms/reduced_39_sandiego_mechCK.yaml',
    )  # ✅ shared state
    """
    # Add species if needed (Only use input_params.species if working with a small chemical mechanism file)
    add_species_vars(input_params.species)

    # Step 3:
    data_dict = data_extraction(transport_species, script_config)

    # Step 4: Transport Budget Analysis
    # Step 4.1: Set output path
    output_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), f'Transport-Budget-Plots')
    os.makedirs(output_path, exist_ok=True)
    # Step 4.2:
    transport_budget_analysis(data_dict, transport_species, output_path, script_config)

    return

if __name__ == "__main__":
    # Step 1: Initialize the script
    start_time = time.time()
    # Logger setup
    init_rank_logging(f'runlog-{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt', overwrite=True)
    # Step 3: Run the main function
    main()
    comm.Barrier()
    # Step 4: Finalize the logger
    if rank == 0:
        end_time = time.time()
        rank_log(f"Script completed successfully. Took {end_time - start_time:.2f} seconds.")
        # Synchronize logs across ranks at the end
        flush_log()