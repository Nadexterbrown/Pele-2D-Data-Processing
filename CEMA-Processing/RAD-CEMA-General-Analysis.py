import os, yt, multiprocessing, re, itertools

import numpy as np

from matplotlib.tri import Triangulation
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

import cantera as ct

from sdtoolbox.postshock import CJspeed, PostShock_fr, PostShock_eq
from sdtoolbox.utilities import CJspeed_plot, znd_plot
from sdtoolbox.znd import zndsolve
from sdtoolbox.cv import cvsolve

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
# Function Scripts - Mathematical Processes
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

########################################################################################################################
# Function Scripts - Data Generation (Cantera)
########################################################################################################################

def cantera_flame():

    ###########################################
    # Internal Functions
    ###########################################

    def dict_creation():
        data = {
            'Grid': f.grid,
            'Temperature': f.T,
            'Pressure': f.P,
            'Density': f.density_mass,
            'Viscosity': f.viscosity,
            'Conductivity': f.thermal_conductivity,
            'Heat Release Rate': f.heat_release_rate,
            'Cp': f.cp_mass,
            'X Velocity': f.velocity
        }

        for species in input_params.species:
            # Mass Fractions (Y_k)
            data[f'Y({species})'] = f.Y[gas.species_index(species)]
            # Mixture Averaged Diffusion Coefficients (D_k)
            data[f'D({species})'] = f.mix_diff_coeffs_mass[gas.species_index(species)]
            # Specific Enthalpies (h_k)
            data[f'h({species})'] = f.standard_enthalpies_RT[gas.species_index(species)] * ct.gas_constant * f.T
            # Species Reaction Rates (Net Rate of Production)
            data[f'W({species})'] = f.net_production_rates[gas.species_index(species)] * gas.molecular_weights[gas.species_index(species)]

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

def sdtoolbox_detonation():
    ###########################################
    # Internal Functions
    ###########################################

    def dict_creation():
        data = {
            'Grid': f.grid,
            'Temperature': f.T,
            'Pressure': f.P,
            'Density': f.density_mass,
            'Viscosity': f.viscosity,
            'Conductivity': f.thermal_conductivity,
            'Heat Release Rate': f.heat_release_rate,
            'Cp': f.cp_mass,
            'X Velocity': f.velocity
        }

        for species in input_params.species:
            # Mass Fractions (Y_k)
            data[f'Y({species})'] = f.Y(species)
            # Mixture Averaged Diffusion Coefficients (D_k)
            data[f'Y({species})'] = f.mix_diff_coeffs_mass(species)
            # Specific Enthalpies (h_k)
            data[f'Y({species})'] = f.standard_enthalpies_RT(species) * ct.gas_constant * f.T

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
# Function Scripts - 2D DNS (Pele) Data Processing
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




########################################################################################################################
# Function Scripts - Chemical Explosive Mode Analysis (CEMA) / Reaction/Advection/Diffusion (RAD) Processing
########################################################################################################################

def rad_analysis(data, species='OH', PLT_FLAG=False):

    ###########################################
    # Internal Functions
    ###########################################

    def reaction_terms():
        return data[f'W({species})']

    def advection_terms():
        tmp_arr = data['Density'] * data[f'Y({species})'] * data['X Velocity']
        return np.gradient(tmp_arr) / np.gradient(data['Grid'])

    def diffusion_terms():
        # Step 1: Compute the species mass fraction gradient
        size = len(data['Grid'])
        grad_Y = np.zeros(size)
        for n in range(size - 1):
            grad_Y[n] = (data[f'Y({species})'][n + 1] - data[f'Y({species})'][n]) / (data['Grid'][n + 1] - data['Grid'][n])

        # Step 2: Calculate the diffusion velocity for the species using Fick's law
        diff_vel = -data[f'D({species})'] * grad_Y / data[f'Y({species})']

        # Step 3: Multiply the terms together for differentiation
        tmp_arr = data['Density'] * data[f'Y({species})'] * diff_vel
        return np.gradient(tmp_arr) / np.gradient(data['Grid'])

    ###########################################
    # Main Function
    #
    # The process outlined here originates from discussion in Chen et al.
    # (https://doi.org/10.1016/j.combustflame.2017.09.012)
    #
    # Here we solve for the right hand terms in the species conservation equation
    ###########################################

    # Step 1: Calculate the various terms for comparison
    r = reaction_terms()
    a = advection_terms()
    d = diffusion_terms()

    if PLT_FLAG:
        # Create the main figure and axis
        fig, ax1 = plt.subplots(figsize=(8, 6))

        # Plot on the first y-axis
        ax1.plot(data['Grid'], r, 'r-', label="Reaction")
        ax1.plot(data['Grid'], a, 'r--', label="Advection")
        ax1.plot(data['Grid'], d, 'r-.', label="Diffusion")
        ax1.set_xlim(0.00065, 0.0008)
        ax1.set_xlabel('Grid')
        ax1.set_ylabel('Primary Y-axis')
        ax1.legend(loc='upper left')

        # Create a second y-axis
        ax2 = ax1.twinx()
        ax2.plot(data['Grid'], data[f'Y({species})'], 'k-', label=f"Y({species})")
        ax2.set_ylabel('Secondary Y-axis')
        ax2.legend(loc='upper right')

        # Display the plot
        plt.show()

    return

def cema_terms():

    ###########################################
    # Main Function
    ###########################################

    def diffusion_terms():

        return

    def chemical_terms():

        return

    ###########################################
    # Main Function
    ###########################################



    return

def cema_analysis():
    ###########################################
    # Main Function
    ###########################################

    def jacobian():

        return



    ###########################################
    # Main Function
    ###########################################

    return

########################################################################################################################
# Main Script
########################################################################################################################

def main():

    # Step 1: Script Flags
    SOLVER_TYPE = {
        'Laminar Flame': True,
        'ZND Detonation': False,
        'PeleC 2D Data': False,
    }

    ANALYSIS_MODE = {
        'RAD': True,
        'CEMA': False,
    }

    # Step 2: Set reactant mixture parameters
    initialize_parameters(
        T=503.15,
        P=10.0 * 100000,
        Phi=1.0,
        Fuel='H2',
        nitrogenAmount=0.5 * 3.76,
        mech='../Chemical-Mechanisms/Li-Dryer-H2-mechanism.yaml',
    )

    # Step 3: Create or extract data from file, depending on SOLVER_TYPE

    if sum(value is True for value in SOLVER_TYPE.values() if isinstance(value, bool)) > 1:
        raise ValueError('Please Select only one data method to be processed')

    if SOLVER_TYPE.get('Laminar Flame', False):
        data = cantera_flame()

    if SOLVER_TYPE.get('ZND Detonation', False):
        data = sdtoolbox_detonation()

    if SOLVER_TYPE.get('PeleC 2D Data', False):
        data = sdtoolbox_detonation()

    # Step 4: Process the data with the stated ANALYSIS_MODE
    if ANALYSIS_MODE.get('RAD', False):
        rad_analysis(data, PLT_FLAG=True)

    if ANALYSIS_MODE.get('CEMA', False):
        cema_analysis(data)

    return

if __name__ == '__main__':
    main()
