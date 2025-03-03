import copy
import os, yt, multiprocessing, re, itertools

import numpy as np

from scipy.interpolate import RegularGridInterpolator, LinearNDInterpolator, griddata
from scipy.linalg import eig, pinv

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
        elif self.Fuel == "CH4":
            self.oxygenAmount = 2.0
        elif self.Fuel == "C2H6":
            self.oxygenAmount = 3.5
        elif self.Fuel == "ch3och3":
            self.oxygenAmount = 3
        elif self.Fuel == "C4H10":
            self.oxygenAmount = 6.5
        else:
            raise ValueError(f"Unknown fuel type: {self.Fuel}")

        # Update the composition dictionary
        try:
            self.X = {
                self.Fuel: self.Phi,
                'O2': self.oxygenAmount,
                'N2': self.nitrogenAmount
            }
        except:
            print('Using lower case chemical composition')
            self.X = {
                self.Fuel: self.Phi,
                'o2': self.oxygenAmount,
                'n2': self.nitrogenAmount
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
            'Grid': np.array([[x, 0] for x in f.grid]),
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
    f.set_refine_criteria(ratio=3, slope=0.1, curve=0.1)
    # Step 4: Solve with mixture-averaged transport model
    f.transport_model = 'mixture-averaged'
    f.solve(loglevel=loglevel, auto=True)

    return dict_creation()

def sdtoolbox_detonation():
    ###########################################
    # Internal Functions
    ###########################################

    def dict_creation():
        # Initialize an empty dictionary with NumPy arrays
        data = {
            'Grid': np.zeros((len(out['distance']), 2)),
            'Temperature': np.zeros(len(out['distance'])),
            'Pressure': np.zeros(len(out['distance'])),
            'Density': np.zeros(len(out['distance'])),
            'Viscosity': np.zeros(len(out['distance'])),
            'Conductivity': np.zeros(len(out['distance'])),
            'Heat Release Rate': np.zeros(len(out['distance'])),
            'Cp': np.zeros(len(out['distance'])),
            'X Velocity': np.zeros(len(out['distance']))
        }

        # Add species-dependent properties dynamically
        for species in input_params.species:
            data[f'Y({species})'] = np.zeros(len(out['distance']))
            data[f'D({species})'] = np.zeros(len(out['distance']))
            data[f'h({species})'] = np.zeros(len(out['distance']))
            data[f'W({species})'] = np.zeros(len(out['distance']))

        # Create a gas object
        gas_tmp = ct.Solution(input_params.mech)

        for i in range(len(out['distance'])):
            # Set gas state for the current step
            species_dict = {input_params.species[j]: out['species'][j][i] for j in range(len(input_params.species))}
            gas_tmp.TPY = out['T'][i], out['P'][i], species_dict

            # Store computed values directly into the NumPy arrays
            data['Grid'][i] = out['distance'][i]
            data['Temperature'][i] = out['T'][i]
            data['Pressure'][i] = out['P'][i]
            data['Density'][i] = gas_tmp.density_mass
            data['Viscosity'][i] = gas_tmp.viscosity
            data['Conductivity'][i] = gas_tmp.thermal_conductivity
            data['Heat Release Rate'][i] = gas_tmp.heat_release_rate
            data['Cp'][i] = gas_tmp.cp_mass
            data['X Velocity'][i] = out['U'][i]

            for species in input_params.species:
                idx = gas_tmp.species_index(species)
                data[f'Y({species})'][i] = gas_tmp.Y[idx]  # Mass Fraction
                data[f'D({species})'][i] = gas_tmp.mix_diff_coeffs_mass[idx]  # Diffusion Coefficient
                data[f'h({species})'][i] = gas_tmp.standard_enthalpies_RT[idx] * ct.gas_constant * gas_tmp.T  # Enthalpy
                data[f'W({species})'][i] = gas_tmp.net_production_rates[idx] * gas_tmp.molecular_weights[idx]  # Reaction Rate

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
    x_bnds, y_bnds, y_loc, ANALYSIS_MODE = const_arr

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

    # Get unique x values and store them in an array
    x_arr = np.unique(x_values)

    x_min = x_arr[np.argmin(np.abs(x_arr - x_bnds[0]))]
    x_max = x_arr[np.argmin(np.abs(x_arr - x_bnds[1]))]
    x_arr = x_arr[(x_arr >= x_min) & (x_arr <= x_max)]

    x_pts = round((x_max - x_min) / dx) + 1

    x_index_map = {x: i for i, x in enumerate(x_arr)}

    # Get unique y values and store them in an array
    y_arr = np.unique(y_values)

    # Update the y bounds to be a valid grid point
    y_min = y_arr[np.argmin(np.abs(y_arr - y_bnds[0]))]
    y_max = y_arr[np.argmin(np.abs(y_arr - y_bnds[1]))]
    y_arr = y_arr[(y_arr >= y_min) & (y_arr <= y_max)]

    # Set the number of points present between the min and max values
    y_pts = round((y_max - y_min) / dx) + 1

    # Create mapping of physical coordinates to array indices
    y_index_map = {y: i for i, y in enumerate(y_arr)}

    # Step 4: Create an empty dictionary for each variable and place an empty object array with length equal to the
    # number of y-values

    target_str = {
        'Grid': 'grid',
        'Temperature': 'Temp',
        'Pressure': 'pressure',
        'Density': 'density',
        'Viscosity': 'viscosity',
        'Conductivity': 'conductivity',
        'Heat Release Rate': 'heatRelease',
        'Cp': 'cp',
        'X Velocity': 'x_velocity',
        'Y Velocity': 'y_velocity'
    }

    for species in input_params.species:
        target_str[f'Y({species})'] = f'Y({species})'
        target_str[f'D({species})'] = f'D({species})'
        target_str[f'W({species})'] = f'rho_omega_{species}'

    if ANALYSIS_MODE.get('CEMA', False):
        data_dict = {
            key: np.full((y_pts, x_pts), np.nan)
            for key, raw_var in target_str.items()
            if raw_var in np.array(raw_data.field_list)[:, 1]  # Directly check existence
        }

        # data_dict['x'] = np.full((y_pts, x_pts), np.nan)  # Initialize with NaNs
        # data_dict['y'] = np.full((y_pts, x_pts), np.nan)  # Initialize with NaNs
        data_dict['Grid'] = np.full((y_pts, x_pts, 2), np.nan, dtype=object)  # Initialize with NaNs
        data_dict['pltFile'] = pltFile_dir
    else:
        y_idx = np.nanargmin(abs(y_arr - y_loc))

        data_dict = {
            key: np.full((x_pts), np.nan)
            for key, raw_var in target_str.items()
            if raw_var in np.array(raw_data.field_list)[:, 1]  # Directly check existence
        }

        # data_dict['x'] = np.full((y_pts, x_pts), np.nan)  # Initialize with NaNs
        # data_dict['y'] = np.full((y_pts, x_pts), np.nan)  # Initialize with NaNs
        data_dict['Grid'] = np.full((x_pts, 2), np.nan, dtype=object)  # Initialize with NaNs
        data_dict['pltFile'] = pltFile_dir

    # Step 5:
    for grid in grids:
        x_vals = grid['boxlib', 'x'].to_value().flatten()
        y_vals = grid['boxlib', 'y'].to_value().flatten()

        for i in range(len(x_vals)):
            x, y = x_vals[i], y_vals[i]
            if ANALYSIS_MODE.get('CEMA', False):
                if x in x_index_map.keys() and y in y_index_map.keys():  # Ensure valid points
                    xi, yi = x_index_map[x], y_index_map[y]  # Get array indices
                    # data_dict['x'][yi, xi] = x
                    # data_dict['y'][yi, xi] = y
                    data_dict['Grid'][yi, xi, :] = (x, y)
                    for key, var in target_str.items():
                        if key not in ('Grid', 'pltFile'):
                            data_dict[key][yi, xi] = grid["boxlib", var].to_value().flatten()[i]  # Place value at the correct index
            else:
                if x in x_index_map.keys() and y in y_index_map.keys():
                    if y_index_map[y] == y_idx:
                        xi = x_index_map[x]  # Get array indices
                        data_dict['Grid'][xi, :] = (x, y)
                        for key, var in target_str.items():
                            if key not in ('Grid', 'pltFile'):
                                data_dict[key][xi] = grid["boxlib", var].to_value().flatten()[
                                    i]  # Place value at the correct index
    return data_dict

def flame_box(data_info, flame_y_loc, box_size):

    # Step 1: Calculate the flame contour
    contour_pts = flame_contour(data_info)

    # Step 2: Determine the flame index
    flame_idx = np.argmin(np.abs(contour_pts[:, 1] - flame_y_loc))
    flame_loc = [contour_pts[flame_idx, 0], contour_pts[flame_idx, 1]]

    x_bnds = [flame_loc[0] - box_size, flame_loc[0] + box_size]
    y_bnds = [flame_loc[1] - box_size, flame_loc[1] + box_size]

    return x_bnds, y_bnds

def flame_contour(data_info, flame_bnds=None, tracking_str='Temp', tracking_val=2000):
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

    # Step 3: Only select points near the flame (remove leading shock points)
    flame_vert = flame_vert[np.abs(np.mean(flame_vert[:, 0]) - flame_vert[:, 0]) <= 1]

    # Step 4: If bounds are given mask the contour
    if flame_bnds:
        # Box flame contour
        mask_x = (flame_vert[:, 0] >= flame_bnds[0][0]) & (flame_vert[:, 0] <= flame_bnds[0][-1])
        mask_y = (flame_vert[:, 1] >= flame_bnds[1][0]) & (flame_vert[:, 1] <= flame_bnds[1][-1])
        mask = mask_x & mask_y

        flame_vert = flame_vert[mask]

    return flame_vert

def cantera_enthalpy(data):

    # Map the temperature and pressure to arrays, and collect the species present in the solution
    temperature = data['Temperature']
    pressure = data['Pressure']
    species_list = [var for var in input_params.species]
    y_pts, x_pts = data['Grid'].shape

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


                    species_enthalpy[f"Y({species})"][j, i] = gas_tmp.standard_enthalpies_RT[gas_tmp.species_index(species)] * ct.gas_constant * temperature[j,i] * 10000

    return species_enthalpy

########################################################################################################################
# Function Scripts - Chemical Explosive Mode Analysis (CEMA) / Reaction/Advection/Diffusion (RAD) Processing
########################################################################################################################

def rad_analysis(data, species='OH', flame_y_loc=None, PLT_FLAG=False):

    ###########################################
    # Internal Functions
    ###########################################

    def reaction_terms():
        return data[f'W({species})']

    def advection_terms():
        tmp_arr = data['Density'] * data[f'Y({species})'] * data['X Velocity']
        return - np.gradient(tmp_arr) / np.gradient(data['Grid'][:, 0])

    def diffusion_terms():
        # Step 1: Compute the species mass fraction gradient
        size = len(data['Grid'])
        grad_Y = np.zeros(size)
        for n in range(size - 1):
            grad_Y[n] = (data[f'Y({species})'][n + 1] - data[f'Y({species})'][n]) / (data['Grid'][n + 1, 0] - data['Grid'][n, 0])

        # Step 2: Calculate the diffusion velocity for the species using Fick's law
        diff_vel = - data[f'D({species})'] * grad_Y / data[f'Y({species})']

        # Replace NaNs/Infs with nearest valid values or 0
        diff_vel[np.isnan(diff_vel) | np.isinf(diff_vel)] = 0

        # Step 3: Multiply the terms together for differentiation
        tmp_arr = data['Density'] * data[f'Y({species})'] * diff_vel
        return - np.gradient(tmp_arr) / np.gradient(data['Grid'][:, 0])

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
        ax1.plot(data['Grid'][:, 0], r / np.linalg.norm(np.nan_to_num(r)), 'r-', label="Reaction")
        ax1.plot(data['Grid'][:, 0], a / np.linalg.norm(np.nan_to_num(a)), 'b--', label="Advection")
        ax1.plot(data['Grid'][:, 0], d / np.linalg.norm(np.nan_to_num(d)), 'g:', label="Diffusion")
        #ax1.set_xlim(0.0006, 0.0008)
        ax1.set_xlabel('Grid')
        ax1.set_ylabel('Value / Norm(Value)')
        ax1.legend(loc='lower left')

        # Create a second y-axis
        ax2 = ax1.twinx()
        ax2.plot(data['Grid'][:, 0], data[f'Y({species})'], 'k-', label=f"Y({species})")
        ax2.set_ylabel('Y(OH)')
        ax2.legend(loc='l right')

        # Display the plot
        plt.show()

    return

def cema_terms(data):

    ###########################################
    # Main Function
    #
    # The process outlined here originates from discussion in Ren et al.
    # https://doi.org/10.2514/1.J057994
    #
    # Here we solve for the Jacobian of the modified energy equation:
    # Dgω(φ)/Dt = J ⋅ Dφ/Dt = J ⋅ gω(φ) + gf; J = ∂gω/∂φ
    ###########################################

    def diffusion_terms():
        # Following the approach from Ren et al., here we calculate the components of the diffusion source terms
        # (g_d, g_r, g_T)

        # Step 1: Pre-allocate space for the arrays
        g_d = np.full((y_pts, x_pts, len(input_params.species) + 3), np.nan)
        g_r = np.full((y_pts, x_pts, len(input_params.species) + 3), np.nan)
        g_T = np.full((y_pts, x_pts, len(input_params.species) + 3), np.nan)

        # Step 2: Pre-calculate the various parameters used in g_d, g_r, g_T
        diffusion_coeffs = [f"D({var})" for var in input_params.species]

        mass_fractions = [f"Y({var})" for var in input_params.species]

        dYdx = {
            var: np.full((y_pts, x_pts), np.nan)  # Initialize with NaNs
            for var in mass_fractions
        }
        dYdy = copy.deepcopy(dYdx)

        Jx = {
            var: np.full((y_pts, x_pts), np.nan)  # Initialize with NaNs
            for var in mass_fractions
        }
        Jy = copy.deepcopy(Jx)

        dJdx = {
            var: np.full((y_pts, x_pts), np.nan)  # Initialize with NaNs
            for var in mass_fractions
        }
        dJdy = copy.deepcopy(dJdx)

        # Calculate the species dependent derivatives (dY, dJ)
        for i, species in enumerate(input_params.species):
            # Species Mass Fraction Derivatives
            dYdx[f'Y({species})'] = spatial_derivative(data[f'Y({species})'], dx, 'x')
            dYdy[f'Y({species})'] = spatial_derivative(data[f'Y({species})'], dy, 'y')
            # Diffusion Flux
            Jx[f'Y({species})'] = data['Density'] * data[diffusion_coeffs[i]] * dYdx[f'Y({species})']
            Jy[f'Y({species})'] = data['Density'] * data[diffusion_coeffs[i]] * dYdy[f'Y({species})']
            # Diffusion Flux Derivatives
            dJdx[f'Y({species})'] = spatial_derivative(Jx[f'Y({species})'], dx, 'x')
            dJdy[f'Y({species})'] = spatial_derivative(Jy[f'Y({species})'], dy, 'y')

        # Calculate the velocity spatial derivatives
        dudx = spatial_derivative(data['X Velocity'], dx, 'x')
        dudy = spatial_derivative(data['X Velocity'], dy, 'y')
        dvdx = spatial_derivative(data['Y Velocity'], dx, 'x')
        dvdy = spatial_derivative(data['Y Velocity'], dy, 'y')

        # Step 4: Calculate the pressure terms (pressure, shear stress)
        dPdx = spatial_derivative(data['Pressure'], dx, 'x')
        dPdy = spatial_derivative(data['Pressure'], dy, 'y')

        dPudx = spatial_derivative(data['Pressure'] * data['X Velocity'], dx, 'x')
        dPvdy = spatial_derivative(data['Pressure'] * data['Y Velocity'], dy, 'y')

        tau_xx = data['Viscosity'] * (dudx + dudx)
        tau_xy = data['Viscosity'] * ((dudy + dvdx) - (2 / 3) * (dudx + dvdy))
        tau_yx = data['Viscosity'] * ((dvdx + dudy) - (2 / 3) * (dudx + dvdy))
        tau_yy = data['Viscosity'] * (dvdy + dvdy)

        dtau_1 = spatial_derivative(tau_xx, dx, 'x') + spatial_derivative(tau_xy, dy, 'y')
        dtau_2 = spatial_derivative(tau_yx, dx, 'x') + spatial_derivative(tau_yy, dy, 'y')

        species_enthalpy = cantera_enthalpy(data)
        g_d_energ_1 = data['Conductivity'] * spatial_derivative(data['Temperature'], dx, 'x') + sum(data['Density'] * data[f'D({species})'] * species_enthalpy[f'Y({species})'] * dYdx[f'Y({species})'] for species in input_params.species) + data['X Velocity'] * tau_xx + data['Y Velocity'] * tau_yx
        g_d_energ_2 = data['Conductivity'] * spatial_derivative(data['Temperature'], dy, 'y') + sum(data['Density'] * data[f'D({species})'] * species_enthalpy[f'Y({species})'] * dYdy[f'Y({species})'] for species in input_params.species) + data['X Velocity'] * tau_xy + data['Y Velocity'] * tau_yy
        dg_d_energ_1_dx = spatial_derivative(g_d_energ_1, dx, 'x')
        dg_d_energ_2_dy = spatial_derivative(g_d_energ_2, dy, 'y')

        # Step 2: Calculate the full spatial array space
        for j in range(0, y_pts):
            for i in range(0, x_pts):
                # Calculate the species terms
                for idx, species in enumerate(input_params.species):
                    g_d[j, i, idx] = dJdx[f'Y({species})'][j, i] + dJdy[f'Y({species})'][j, i]
                    g_r[j, i, idx] = -data[f'rho_{species}'][j, i] * (dudx[j, i] + dvdy[j, i])
                    if not np.isnan(g_r[j, i, idx]):
                        g_T[j, i, idx] = 0

                # Calculate the advection terms
                g_d[j, i, -3] = dtau_1[j, i]
                g_d[j, i, -2] = dtau_2[j, i]

                g_r[j, i, -3] = -data['Density'][j, i] * data['X Velocity'][j, i] * (dudx[j, i] + dvdy[j, i])
                g_r[j, i, -2] = -data['Density'][j, i] * data['Y Velocity'][j, i] * (dudx[j, i] + dvdy[j, i])

                g_T[j, i, -3] = -dPdx[j, i]
                g_T[j, i, -2] = -dPdy[j, i]

                # Calculate the energy terms
                g_d[j, i, -1] = dg_d_energ_1_dx[j, i] + dg_d_energ_2_dy[j, i]

                g_r[j, i, -1] = -data['rho_e'][j, i] * (dudx[j, i] + dvdy[j, i])

                g_T[j, i, -1] = -(dPudx[j, i] + dPvdy[j, i])

        return g_d, g_r, g_T

    def chemical_terms():
        # Step 1: Pre-allocate space for the arrays
        g_w = np.full((y_pts, x_pts, len(input_params.species) + 3), np.nan)

        # Step 3: Calculate the full spatial array space
        for j in range(0, y_pts):
            for i in range(0, x_pts):
                # Calculate the species terms
                for idx, species in enumerate(input_params.species):
                    g_w[j, i, idx] = data[f'W({species})'][j, i] #/ data['density'][j, i]

                if not np.isnan(g_w[j, i, 0]):
                    # Calculate the advection terms
                    g_w[j, i, -3] = 0
                    g_w[j, i, -2] = 0

                    # Calculate the energy terms
                    g_w[j, i, -1] = 0  # Skip NaN values

        return g_w

    ###########################################
    # Main Function
    ###########################################

    # Step 1: Extract grid spacing from domain (constant grid spacing at set level)
    y_pts, x_pts = data['Grid'].shape
    dx = dy = np.diff(data['Grid'][:, :, 0], axis=1)[~np.isnan(np.diff(data['Grid'][:, :, 0], axis=1))][0]

    # Step 2: Calculate the diffusion terms for CEMA analysis
    g_d, g_r, g_T = diffusion_terms()

    # Step 3: Calculate the reaction terms for CEMA analysis
    g_w = chemical_terms()

    return g_w, g_d, g_r, g_T

def cema_analysis(data):
    ###########################################
    # Main Function
    ###########################################

    def jacobian():
        """
            Compute the Jacobian dg/dp for each (x, y) point.

            Parameters:
                g_w: ndarray of shape (x_pts, y_pts, z) - Function values at each (x, y)
                state_vec: ndarray of shape (x_pts, y_pts, z) - Parameter values at each (x, y)

            Returns:
                J: ndarray of shape (x_pts, y_pts, z, z) - Jacobian at each (x, y)
            """
        x_pts, y_pts, z = g_w.shape
        dx = np.diff(data['Grid'][:, :, 0], axis=1)[~np.isnan(np.diff(data['Grid'][:, :, 0], axis=1))][0]
        J = np.zeros((x_pts, y_pts, z, z))  # Initialize Jacobian storage

        for i in range(z):  # Loop over output variables (g_w components)
            for j in range(z):  # Loop over input variables (state_vec components)
                # J[:, :, i, j] = np.gradient(g_w[:, :, i], edge_order=2) / np.gradient(state_vec[:, :, j], edge_order=2)
                grad_gw_x, grad_gw_y = np.gradient(g_w[:, :, i], dx, edge_order=2)
                grad_state_vec_x, grad_state_vec_y = np.gradient(state_vec[:, :, j], dx, edge_order=2)

                # Combine the gradients into their magnitude (Euclidean norm)
                grad_gw = np.sqrt(grad_gw_x ** 2 + grad_gw_y ** 2)
                grad_state_vec = np.sqrt(grad_state_vec_x ** 2 + grad_state_vec_y ** 2)

                # Now compute the element-wise ratio
                J[:, :, i, j] = grad_gw / grad_state_vec

        return J

    ###########################################
    # Main Function
    ###########################################

    y_pts, x_pts = data['x'].shape

    # Step 1: Determine the source term vectors
    g_w, g_d, g_r, g_T = cema_terms(data)

    # Step 1: Create the state vector and calculate the jacobian at each point in the domain
    state_vec = np.zeros((y_pts, x_pts, len(input_params.species) + 3))
    for j in range(0, y_pts):
        for i in range(0, x_pts):
            # Calcualte the state vector
            for idx, species in enumerate(input_params.species):
                state_vec[j, i, idx] = data['Density'][j, i] * data[f'Y({species})'][j, i]
            state_vec[j, i, -3] = data['Density'][j, i] * data['X Velocity'][j, i]
            state_vec[j, i, -2] = data['Density'][j, i] * data['Y Velocity'][j, i]
            state_vec[j, i, -1] = data['rho_e'][j, i]

    # Calculate the Jacobian
    J = jacobian()

    # Step 2: Determine the eigenstate at each point
    eig_val = np.zeros((y_pts, x_pts, len(input_params.species) + 3), dtype=np.complex128)
    eig_vec = np.zeros((y_pts, x_pts, len(input_params.species) + 3, len(input_params.species) + 3), dtype=np.complex128)

    for j in range(0, y_pts):
        for i in range(0, x_pts):
            J_local = J[j, i]
            if np.isnan(J_local).any():
                continue  # Skip NaN values
            eig_val[j, i], eig_vec[j, i], _ = eig(J_local, left=True)

    phi_w = np.full((y_pts, x_pts), np.nan)
    phi_d = np.full((y_pts, x_pts), np.nan)
    phi_r = np.full((y_pts, x_pts), np.nan)
    phi_T = np.full((y_pts, x_pts), np.nan)
    phi_f = np.full((y_pts, x_pts), np.nan)
    alpha = np.full((y_pts, x_pts), np.nan)
    for j in range(0, y_pts):
        for i in range(0, x_pts):
            # Step 2: Determine the mode with the largest positive real part eigenvalue
            if np.isnan(eig_val[j, i, :]).any():
                continue  # Skip NaN values

            local_val = eig_val[j, i, :]
            local_vec = eig_vec[j, i, :, :]

            valid_mask = local_val.real > 0
            if (local_val[valid_mask]).size == 0:
                continue

            eig_idx = np.argmax(local_val[valid_mask])
            lambda_e = (local_val[valid_mask])[eig_idx]
            b_e = (local_vec[:, valid_mask].real)[:, eig_idx]

            phi_w[j, i] = np.dot(b_e, g_w[j, i])
            phi_d[j, i] = np.dot(b_e, g_d[j, i])
            phi_r[j, i] = np.dot(b_e, g_r[j, i])
            phi_T[j, i] = np.dot(b_e, g_T[j, i])
            phi_f[j, i] = np.dot(b_e, g_d[j, i] + g_r[j, i] + g_T[j, i])
            alpha[j, i] = phi_f[j, i] / phi_w[j, i]

    return phi_w, phi_f, alpha

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
        T=503,
        P=10.0 * 1e5,
        Phi=1.0,
        Fuel='H2',
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
        # Collect all the present PeleC data directories
        dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../')

        time_data_dir = [os.path.join(dir_path, raw_data_folder, time_step)
                         for raw_data_folder in os.listdir(dir_path)
                         if os.path.isdir(os.path.join(dir_path, raw_data_folder)) and raw_data_folder.startswith(
                f'{data_set}')
                         for time_step in os.listdir(os.path.join(dir_path, raw_data_folder))
                         if os.path.isdir(os.path.join(dir_path, raw_data_folder, time_step)) and time_step.startswith(
                'plt')]

        # Chronologically order the pltFiles and truncate the raw data list if skip loading is enabled
        updated_data_list = sort_files(time_data_dir)
        updated_data_list = [updated_data_list[0]]

        # Step 3:
        ddt_box_size = 1e-2  # cm
        ddt_y_loc = 0.0462731
        input_params.species.remove("N2")
        # If multiple plot files are provided, determine the size of the box to fit data from the first and last file
        if len(np.array(updated_data_list)) > 1:
            initial_box = flame_box(updated_data_list[0], flame_y_loc=ddt_y_loc, box_size=ddt_box_size)
            final_box = flame_box(updated_data_list[-1], flame_y_loc=ddt_y_loc, box_size=ddt_box_size)
        else:
            initial_box = final_box = flame_box(updated_data_list[0], flame_y_loc=ddt_y_loc, box_size=ddt_box_size)

        x_bnds, y_bnds = [np.minimum(initial_box[0], final_box[0]), np.maximum(initial_box[1], final_box[1])]

        # Extract the data from the plot files in parallel
        print('Beginning Data Import')
        collective_data =  parallel_processing_function(updated_data_list, (x_bnds, y_bnds, ddt_y_loc, ANALYSIS_MODE,), pelec_flame_data)
        print('Completed Data Import')

    # Step 4: Process the data with the stated ANALYSIS_MODE
    if ANALYSIS_MODE.get('RAD', False):
        if SOLVER_TYPE.get('PeleC 2D Data', False):
            for data in collective_data:
                rad_analysis(data, PLT_FLAG=True)
        else:
            rad_analysis(data, PLT_FLAG=True)

    if ANALYSIS_MODE.get('CEMA', False):
        for data in collective_data:
            phi_w, phi_f, alpha = cema_analysis(data)

            progress_var = 0.05
            progress_var_pts = flame_contour(data, (x_bnds, y_bnds), tracking_str='Y(H2O)',
                                             tracking_val=progress_var * (np.nanmax(data['Y(H2O)']) - np.nanmin(data['Y(H2O)'])) + np.nanmin(data['Y(H2O)']))

            x_arr = np.unique(data['x'])[~np.isnan(np.unique(data['x']))]
            y_arr = np.unique(data['y'])[~np.isnan(np.unique(data['y']))]
            flame_alpha = []
            masked_alpha = np.full_like(alpha, np.nan)
            for idx in range(len(progress_var_pts)):
                x_idx = int(np.nanargmin(abs(x_arr - progress_var_pts[idx, 0])))
                y_idx = int(np.nanargmin(abs(y_arr - progress_var_pts[idx, 1])))

                flame_alpha.append(alpha[y_idx, x_idx])
                masked_alpha[y_idx, x_idx] = alpha[y_idx, x_idx]

            plt.figure(figsize=(8, 6))
            cmap = mcolors.LinearSegmentedColormap.from_list("custom_rgb", ["red", "red", "blue", "green", "green"],
                                                             N=256)
            norm = mcolors.LogNorm(vmin=0.1, vmax=1000)
            # Use pcolormesh for uniform grids
            mesh = plt.pcolormesh(x_arr, y_arr, abs(masked_alpha), shading='auto', cmap=cmap, norm=norm)
            cbar = plt.colorbar(mesh)
            cbar.set_label("alpha")
            plt.show()

            plt.figure(figsize=(8, 6))
            cmap = mcolors.LinearSegmentedColormap.from_list("custom_rgb", ["red", "red", "blue", "green", "green"],
                                                             N=256)
            norm = mcolors.LogNorm(vmin=0.1, vmax=1000)
            # Use pcolormesh for uniform grids
            mesh = plt.pcolormesh(x_arr, y_arr, abs(alpha), shading='auto', cmap=cmap, norm=norm)
            cbar = plt.colorbar(mesh)
            cbar.set_label("alpha")
            plt.show()

            plt.figure(figsize=(8, 6))
            # Use pcolormesh for uniform grids
            mesh = plt.pcolormesh(x_arr, y_arr, phi_f, shading='auto', cmap='hot')
            cbar = plt.colorbar(mesh)
            cbar.set_label("phi_f")
            plt.show()

            plt.figure(figsize=(8, 6))
            # Use pcolormesh for uniform grids
            mesh = plt.pcolormesh(x_arr, y_arr, phi_w, shading='auto', cmap='hot')
            cbar = plt.colorbar(mesh)
            cbar.set_label("phi_w")
            plt.show()

    return

if __name__ == '__main__':
    main()
