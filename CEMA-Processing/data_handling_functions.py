# Import Libraries
import os, yt, multiprocessing, re, itertools

import cantera as ct
import numpy as np

from scipy.interpolate import griddata

from matplotlib.tri import Triangulation
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from sdtoolbox.postshock import CJspeed, PostShock_fr, PostShock_eq
from sdtoolbox.utilities import CJspeed_plot
from sdtoolbox.znd import zndsolve

########################################################################################################################
# Function Scripts - Data Generation (Cantera)
########################################################################################################################

def cantera_flame(input_params, width=0.001, spacing=0.0001):

    ###########################################
    # Internal Functions
    ###########################################

    def dict_creation():
        # Initialize an empty dictionary with NumPy arrays
        data = {
            'Dimension': 1,
            'Grid': np.zeros((1, len(f.grid), 2)),
            'Temperature': np.zeros((1, len(f.grid))),
            'Pressure': np.zeros((1, len(f.grid))),
            'Density': np.zeros((1, len(f.grid))),
            'Viscosity': np.zeros((1, len(f.grid))),
            'Conductivity': np.zeros((1, len(f.grid))),
            'Heat Release Rate': np.zeros((1, len(f.grid))),
            'Cp': np.zeros((1, len(f.grid))),
            'Cv': np.zeros((1, len(f.grid))),
            'X Velocity': np.zeros((1, len(f.grid))),
            'rho_e': np.zeros((1, len(f.grid)))
        }

        # Add species-dependent properties dynamically
        for species in input_params.species:
            data[f'Y({species})'] = np.zeros((1, len(f.grid)))
            data[f'D({species})'] = np.zeros((1, len(f.grid)))
            data[f'h({species})'] = np.zeros((1, len(f.grid)))
            data[f'W({species})'] = np.zeros((1, len(f.grid)))

        # Store computed values directly into the NumPy arrays
        data['Grid'][0, :, 0] = f.grid
        data['Temperature'][0, :] = f.T
        data['Pressure'][0, :] = f.P * np.ones(len(f.grid))
        data['Density'][0, :] = f.density_mass
        data['Viscosity'][0, :] = f.viscosity
        data['Conductivity'][0, :] = f.thermal_conductivity
        data['Heat Release Rate'][0, :] = f.heat_release_rate
        data['Cp'][0, :] = f.cp_mass
        data['Cv'][0, :] = f.cv_mass
        data['X Velocity'][0, :] = f.velocity
        data['rho_e'][0, :] = f.density_mass * f.int_energy_mass

        for species in input_params.species:
            # Mass Fractions (Y_k)
            data[f'Y({species})'][0, :] = f.Y[gas.species_index(species)]
            # Mixture Averaged Diffusion Coefficients (D_k)
            data[f'D({species})'][0, :] = f.mix_diff_coeffs_mass[gas.species_index(species)]
            # Specific Enthalpies (h_k)
            data[f'h({species})'][0, :] = f.standard_enthalpies_RT[gas.species_index(species)] * ct.gas_constant * f.T
            # Species Reaction Rates (Net Rate of Production)
            data[f'W({species})'][0, :] = f.net_production_rates[gas.species_index(species)] * gas.molecular_weights[gas.species_index(species)]

        return data

    ###########################################
    # Main Function
    ###########################################

    # Step 1: Simulation parameters
    loglevel = 0  # amount of diagnostic output (0 to 8)
    grid = np.arange(0, width + spacing, spacing)
    # Step 2: Solution object used to compute mixture properties, set to the state of the
    #         upstream fuel-air mixture
    gas = ct.Solution(input_params.mech)
    gas.TPX = input_params.T, input_params.P, input_params.X
    # Step 3: Set up flame object
    # f = ct.FreeFlame(gas, grid=grid)
    f = ct.FreeFlame(gas, width=width)
    f.set_refine_criteria(ratio=3, slope=0.1, curve=0.1)
    # Step 4: Solve with mixture-averaged transport model
    f.transport_model = 'mixture-averaged'
    f.solve(loglevel=loglevel, refine_grid=True, auto=True)

    res = dict_creation()
    return res

def sdtoolbox_detonation(input_params):
    ###########################################
    # Internal Functions
    ###########################################

    def dict_creation():
        # Initialize an empty dictionary with NumPy arrays
        data = {
            'Dimension': 1,
            'Grid': np.zeros((1, len(out['distance']), 2)),
            'Temperature': np.zeros((1, len(out['distance']))),
            'Pressure': np.zeros((1, len(out['distance']))),
            'Density': np.zeros((1, len(out['distance']))),
            'Viscosity': np.zeros((1, len(out['distance']))),
            'Conductivity': np.zeros((1, len(out['distance']))),
            'Heat Release Rate': np.zeros((1, len(out['distance']))),
            'Cp': np.zeros((1, len(out['distance']))),
            'Cv': np.zeros((1, len(out['distance']))),
            'X Velocity': np.zeros((1, len(out['distance']))),
            'Y Velocity': np.zeros((1, len(out['distance']))),
            'rho_e': np.zeros((1, len(out['distance'])))
        }

        # Add species-dependent properties dynamically
        for species in input_params.species:
            data[f'Y({species})'] = np.zeros((1, len(out['distance'])))
            data[f'D({species})'] = np.zeros((1, len(out['distance'])))
            data[f'h({species})'] = np.zeros((1, len(out['distance'])))
            data[f'W({species})'] = np.zeros((1, len(out['distance'])))

        data['Grid'][0, :, 0] = out['distance']
        data['Temperature'][0, :] = out['T']
        data['Pressure'][0, :] = out['P']
        data['X Velocity'][0, :] = out['U']

        # Create a gas object
        gas_tmp = ct.Solution(input_params.mech)

        for i in range(len(out['distance'])):
            # Set gas state for the current step
            species_dict = {input_params.species[j]: out['species'][j][i] for j in range(len(input_params.species))}
            gas_tmp.TPY = out['T'][i], out['P'][i], species_dict

            # Store computed values directly into the NumPy arrays
            data['Density'][0, i] = gas_tmp.density_mass
            data['Viscosity'][0, i] = gas_tmp.viscosity
            data['Conductivity'][0, i] = gas_tmp.thermal_conductivity
            data['Heat Release Rate'][0, i] = gas_tmp.heat_release_rate
            data['Cp'][0, i] = gas_tmp.cp_mass
            data['Cv'][0, i] = gas_tmp.cv_mass
            data['rho_e'][0, i] = gas_tmp.density_mass * gas_tmp.int_energy_mass

            for species in input_params.species:
                idx = gas_tmp.species_index(species)
                data[f'Y({species})'][0, i] = gas_tmp.Y[idx]  # Mass Fraction
                data[f'D({species})'][0, i] = gas_tmp.mix_diff_coeffs_mass[idx]  # Diffusion Coefficient
                data[f'h({species})'][0, i] = gas_tmp.standard_enthalpies_RT[idx] * ct.gas_constant * gas_tmp.T  # Enthalpy
                data[f'W({species})'][0, i] = gas_tmp.net_production_rates[idx] * gas_tmp.molecular_weights[idx]  # Reaction Rate

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
    out = zndsolve(gas, gas1, cj_speed, t_end=5e-6, advanced_output=True)

    return dict_creation()



