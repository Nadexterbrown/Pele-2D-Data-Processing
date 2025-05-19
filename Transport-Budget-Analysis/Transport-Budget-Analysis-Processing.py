import re
import os
import yt
import time
import numpy as np
import cantera as ct
from mpi4py import MPI
from ast import literal_eval
from datetime import datetime
from collections import defaultdict
from itertools import groupby, repeat
from scipy.interpolate import CubicSpline
from dataclasses import dataclass, field, fields
from yt.utilities.parallel_tools.parallel_analysis_interface import parallel_objects, communication_system


import matplotlib.pyplot as plt

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
# Flame/Detonation Simulation
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
# 2D Pele Data Extraction
########################################################################################################################


########################################################################################################################
# Math Helper Functions
########################################################################################################################

def grid_regularization(data):
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
    for x_arr in data:
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


def data_interpolation(x, y, x_interp):
    # Create cubic spline with extrapolation disabled
    spline = CubicSpline(x, y, extrapolate=False)

    # Interpolate values (in-range get interpolated, out-of-range become nan)
    y_interp = spline(x_interp)

    # Manually clamp out-of-bounds: fill with first/last y value
    y_interp = np.where(x_interp < x[0], y[0], y_interp)
    y_interp = np.where(x_interp > x[-1], y[-1], y_interp)

    return y_interp


########################################################################################################################
# Transport Budget Analysis (CDR)
########################################################################################################################

def transport_budget_analysis(data, transport_species, data_type, output_path):

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

            try:
                T_tmp = data_interpolation(local_data[i]['Data']['X'], local_data[i]['Data']['Temperature'], x_uniform)
                P_tmp = data_interpolation(local_data[i]['Data']['X'], local_data[i]['Data']['Pressure'], x_uniform)
                Y_tmp = {f'Y({species})': data_interpolation(local_data[i]['Data']['X'], local_data[i]['Data'][f'Y({species})'], x_uniform)
                         for species in input_params.species}
                V_tmp = data_interpolation(local_data[i]['Data']['X'], local_data[i]['Data']['X Velocity'], x_uniform)

                X_tmp = {}
                for species in input_params.species:
                    X_tmp[f'X({species})'] = []

                for idx in range(len(x_uniform)):
                    Y_vec =  {f'{species}': Y_tmp[f'Y({species})'][idx] for species in input_params.species}

                    gas_tmp.TPY = T_tmp[idx], P_tmp[idx], Y_vec
                    for species in input_params.species:
                        X_tmp[f'X({species})'].append(gas_tmp.X[gas_tmp.species_index(species)])

            except Exception:
                T_tmp = data_interpolation(local_data[i]['Data']['X'], local_data[i]['Data']['Temperature'], x_uniform)
                D_tmp = data_interpolation(local_data[i]['Data']['X'], local_data[i]['Data']['Density'], x_uniform)
                Y_tmp = {f'Y({species})': data_interpolation(local_data[i]['Data']['X'], local_data[i]['Data'][f'Y({species})'], x_uniform)
                         for species in input_params.species}
                V_tmp = data_interpolation(local_data[i]['Data']['X'], local_data[i]['Data']['X Velocity'],
                                           x_uniform)

                P_tmp = []
                X_tmp = {}
                for species in input_params.species:
                    X_tmp[f'X({species})'] = []

                for idx in range(len(x_uniform)):
                    Y_vec =  {f'{species}': Y_tmp[f'Y({species})'][idx] for species in input_params.species}

                    gas_tmp.TDY = T_tmp[idx], D_tmp[idx], Y_vec
                    P_tmp.append(gas_tmp.P)
                    for species in input_params.species:
                        X_tmp[f'X({species})'].append(gas_tmp.X[gas_tmp.species_index(species)])

            # Mole Fraction Gradient
            tmp_mole_frac_grad = {f'X({species})': [] for species in input_params.species}
            for k, species in enumerate(input_params.species):
                tmp_mole_frac_grad[f'X({species})'] = np.gradient(X_tmp[f'X({species})'], x_uniform)

            # Flux Calculation
            tmp_mass_flux, tmp_species_flux, tmp_diffusion_flux, tmp_momentum_flux, tmp_energy_flux = [], [], [], [], []
            for idx in range(len(x_uniform)):
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
        # Step 2: Determine the maximum number of time steps in a single rank
        tmp_results = {}
        for sto, i in parallel_objects(data.keys(), njobs=size, storage=tmp_results):
            sto.result_id = int(i)
            sto.result = {
                'Time': data[i]['Time'],
                'Flux': flux_data[i]['Species Flux']
            }

        comm.Barrier()

        if rank == 0:
            print('Hurray!', flush=True)

            # Extract time and flux arrays
            tmp_time_arr = [v['Time'] for v in tmp_results.values()]
            tmp_flux_arr = [v['Flux'] for v in tmp_results.values()]
            time_arr = np.array(tmp_time_arr)
            flux_arr = np.stack(tmp_flux_arr)

            # Ensure time is sorted
            idx = np.argsort(time_arr)
            time_arr = time_arr[idx]
            flux_arr = flux_arr[idx]

            # Compute time derivative
            result = np.gradient(flux_arr, time_arr, axis=0)

            print(result.shape, flush=True)
            return result

        else:
            return None


    ###############################################################################
    # Main Function
    ###############################################################################
    if rank == 0:
        # Extract the spatial grids at each time step
        tmp_x_arr = [v['Data']['X'] for v in data.values()]
        x_uniform = grid_regularization(tmp_x_arr)  # Can still compute dx and x_uniform as before
        print('Uniform X:', x_uniform, flush=True)
        print('Uniform X Length:', len(x_uniform), flush=True)
    else:
        x_uniform = None

    # Broadcast back to all ranks
    x_uniform = comm.bcast(x_uniform, root=0)
    comm.Barrier()

    # Determine all conservation fluxes
    flux_data = flux_calculation()

    # Step 2:
    drhoYdt = unsteady_term()

    # Step 2:
    if rank == 0:
        print(data.keys())
        for i, key in enumerate(data.keys()):
            # Step 2.1:
            tmp_data = data[key]['Data']
            tmp_flux_data = flux_data[key]
            # Step 2.3:
            C = np.gradient(tmp_flux_data['Mass Flux'] * data_interpolation(tmp_data['X'], tmp_data[f'Y({transport_species})'], x_uniform), x_uniform)
            # Step 2.4:
            D = np.gradient(tmp_flux_data['Diffusion Flux'], x_uniform)
            # Step 2.5:
            R = data_interpolation(tmp_data['X'], tmp_data[f'W({transport_species})'], x_uniform)
            # Step 2.6:
            if np.max(R) > abs(np.min(R)):
                total = (drhoYdt[i] + C + D + R)
            else:
                total = (drhoYdt[i] + C + D - R)

            #
            plot_center = x_uniform[np.argmax(data_interpolation(tmp_data['X'], tmp_data['Y(HO2)'], x_uniform))]
            animation_frame_generation(x_uniform, (drhoYdt[i], C, D, R, total),
                                       ('drhoY/dt', 'C', 'D', 'R', 'Total'),
                                       os.path.join(output_path, f'Overall-Plot-{key}.png'),
                                       split_axis=False, plot_center=plot_center, window_size=1e-4)
            animation_frame_generation(x_uniform, data_interpolation(tmp_data['X'], tmp_data['Temperature'], x_uniform),
                                       'Temperature',
                                       os.path.join(output_path, f'Overall-Temperature-Plot-{key}.png'),
                                       split_axis=False)

    return


########################################################################################################################
# Main Script
########################################################################################################################

def main():
    # Step 1:
    dir_path = os.path.dirname(os.path.realpath(__file__))
    sltorc_path = os.path.join(dir_path, '../1D-SLTORC-Data/C2H6-O2/Phi-1.0')
    pele_path = os.path.join(dir_path, '../2D-Test-Data')

    transport_species = 'HO2'

    freeflame_flag = False
    detonation_flag = False
    sltorc_flag = False
    pele_flag = True

    # Step 2: Define the input parameters

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
    if freeflame_flag:
        # if rank == 0:
        # Step 3.1: Extract data
        tmp_data = cantera_flame(transport_species)
        # Step 3.2: Map data to dict
        data_dict = {}
        data_dict[0] = {
            'Time': 0.0,
            'Data': tmp_data
        }
        data_dict[1] = {
            'Time': 1.0,
            'Data': tmp_data
        }
        data_dict[2] = {
            'Time': 2.0,
            'Data': tmp_data
        }
        data_dict[3] = {
            'Time': 3.0,
            'Data': tmp_data
        }

        # Step 3.3: Set output path
        output_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Cantera-H2-Plots')
        os.makedirs(output_path, exist_ok=True)

        # Step 3.4: Perform transport budget analysis
        transport_budget_analysis(data_dict, transport_species, 'cantera', output_path)


    if detonation_flag:
        if rank == 0:
            # W.I.P.
            print('Work in Progress')

    if sltorc_flag:
        # Step 3.1: Extract data
        data_dict = sltorc_data_loading(sltorc_path, transport_species, stride=1)

        output_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'SLTORC-C2H6-Plots')
        os.makedirs(output_path, exist_ok=True)
        # Step 3.3: Perform transport budget analysis
        print(len(data_dict.keys()), flush=True)
        transport_budget_analysis(data_dict, transport_species, 'SLTORC', output_path)

    if pele_flag:
        data_paths = load_directories(pele_path)
        updated_data_paths = sort_files(data_paths)
        pltname_list = [os.path.basename(d) for d in updated_data_paths]

        data_dict = {}
        for sto, dir in yt.parallel_objects(updated_data_paths, -1, storage=data_dict):
            # Step 4.1: Load the data
            ds = yt.load(dir)
            # Step 4.2: Process each file
            sto.result_id = pltname_list.index(ds.basename)
            sto.result = {
                'Time': ds.current_time.to_value(),
                'Data': data_ray_extraction(ds, 0.0445 / 100)
            }

        output_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Pele-2D-Plots')
        os.makedirs(output_path, exist_ok=True)

        transport_budget_analysis(data_dict, transport_species, 'Pele', output_path)

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