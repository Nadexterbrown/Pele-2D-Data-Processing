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

# Import Local Modules
import data_handling_functions
# import pele_functions
import cema_functions

########################################################################################################################
# Global Variables
########################################################################################################################
max_procs = 32
parallel_procs = 16

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
# Function Scripts - Parallel Processing
########################################################################################################################

def worker_function(args):
    """Worker function to process data and return the result."""
    iter_var, const_list, shared_input_params, predicate, kwargs = args
    global input_params
    input_params = shared_input_params
    result = predicate((iter_var, const_list, kwargs))
    return result

def parallel_processing_function(iter_arr, const_list, predicate, n_procs=parallel_procs, **kwargs):
    if n_procs > max_procs:
        print("More processors requested than available. Please update the number of processors requested.")

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
# Function Scripts - 2D DNS (Pele) Data Processing
########################################################################################################################

def process_grid_data(args):
    ###########################################
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

    ###########################################
    # Main Function
    ###########################################
    global input_params
    grid = args[0]
    raw_data, x_index_map, y_index_map, ANALYSIS_MODE, target_str = args[1]

    # Check that all the target strings are in the data file, if not then use cantera to fill the gaps
    missing_str = {key: raw_var for key, raw_var in target_str.items()
                   if raw_var not in np.array(raw_data.field_list)[:, 1] and key != 'Grid'}
    if missing_str:
        gas_missing = ct.Solution(input_params.mech)

    data_dict = {}
    x_vals = grid['boxlib', 'x'].to_value().flatten() * 0.01
    y_vals = grid['boxlib', 'y'].to_value().flatten() * 0.01

    conversion_factor = {
        'Temperature': 1,
        'Pressure': 0.1,
        'Density': 1000,
        'Viscosity': 0.1,
        'Conductivity': 0.00001,
        'Heat Release Rate': 0.1,
        'Cp': 0.0001,
        'Cv': 0.0001,
        'X Velocity': 0.01,
        'Y Velocity': 0.01,
        'rho_e': 0.1,
    }

    for species in input_params.species:
        conversion_factor[f'rho_{species}'] = 1000
        conversion_factor[f'Y({species})'] = 1
        conversion_factor[f'D({species})'] = 0.0001
        conversion_factor[f'W({species})'] = 1000

    for i in range(len(x_vals)):
        x, y = x_vals[i], y_vals[i]
        if x in x_index_map.keys() and y in y_index_map.keys():
            xi, yi = x_index_map[x], y_index_map[y]
            if ANALYSIS_MODE.get('Simple CEMA', False) or ANALYSIS_MODE.get('Compressible CEMA', False):
                data_dict.setdefault('Grid', {}).setdefault((yi, xi), (x, y))
                for key, var in target_str.items():
                    if key not in ('Grid', 'pltFile', 'Dimension'):
                        if key in missing_str:
                            # Only use cantera_str_acquisition if this key is missing
                            missing_dict = cantera_str_acquisition(grid, i, gas_missing)
                            data_dict.setdefault(key, {})[(yi, xi)] = missing_dict.get(key, np.nan)
                        else:
                            # If key is not missing, use existing grid data
                            data_dict.setdefault(key, {})[(yi, xi)] = grid["boxlib", var].to_value().flatten()[i]

    return data_dict


def pelec_flame_data(pltFile_dir, x_bnds, y_bnds, y_loc, ANALYSIS_MODE):
    ###########################################
    # Main Function
    ###########################################
    global input_params

    # Step 1: Load the pelec data
    raw_data = yt.load(pltFile_dir)

    # Step 2: Get the maximum refinement level and smallest grid spacing
    max_level = raw_data.index.max_level
    dx = raw_data.index.get_smallest_dx().to_value() * 0.01

    # Step 3: Extract the desired refinement level grid
    grids = [grid for grid in raw_data.index.grids if grid.Level == max_level]

    # Step 4: Extract all the unique y-values present in the desired refinement level grid
    x_values, y_values = [], []
    for grid in grids:
        x_values.extend(grid['boxlib', 'x'].to_value().flatten() * 0.01)
        y_values.extend(grid['boxlib', 'y'].to_value().flatten() * 0.01)

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

    # Step 4: Create an empty dictionary for each variable
    target_str = {
        'Dimension': 'Dimension',
        'Grid': 'grid',
        'Temperature': 'Temp',
        'Pressure': 'pressure',
        'Density': 'density',
        'Viscosity': 'viscosity',
        'Conductivity': 'conductivity',
        'Heat Release Rate': 'heatRelease',
        'Cp': 'cp',
        'Cv': 'cv',
        'X Velocity': 'x_velocity',
        'Y Velocity': 'y_velocity',
        'rho_e': 'rho_e',
    }

    for species in input_params.species:
        target_str[f'Y({species})'] = f'Y({species})'
        target_str[f'D({species})'] = f'D({species})'
        target_str[f'W({species})'] = f'rho_omega_{species}'
        target_str[f'rho_{species}'] = f'rho_{species}'

    # Step 5: Extract data from the grids in parallel
    individual_grid_data = parallel_processing_function(grids, (raw_data, x_index_map, y_index_map, ANALYSIS_MODE, target_str),
                                                        process_grid_data, n_procs=1)

    # Step 6: Combine results into a single dictionary
    combined_data = {
        key: np.full((y_pts, x_pts), np.nan)
        for key, raw_var in target_str.items()
    }
    combined_data['Grid'] = np.full((y_pts, x_pts, 2), np.nan, dtype=float)  # Initialize with NaNs
    combined_data['pltFile'] = pltFile_dir
    combined_data['Dimension'] = 2

    for grid_data in individual_grid_data:
        for key, values in grid_data.items():
            for coord, value in values.items():
                if key == 'Grid':
                    combined_data[key][coord[0], coord[1], :] = value
                else:
                    combined_data[key][coord[0], coord[1]] = value

    return combined_data

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
    ###########################################
    # Internal Functions
    ###########################################
    def manually_aquire_flame_contour():
        # Get the maximum refinement level
        max_level = raw_data.index.max_level
        # Initialize containers for the highest level data
        x_coords, y_coords, tracking_arr = [], [], []
        # Loop through grids and extract data at the highest level
        for grid in raw_data.index.grids:
            if grid.Level == max_level:
                x_coords.append(grid["boxlib", "x"].to_value().flatten() * 0.01)
                y_coords.append(grid["boxlib", "y"].to_value().flatten() * 0.01)
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

########################################################################################################################
# Main Script
########################################################################################################################

def main():

    # Step 0: Set the data set to be processed
    data_set = '2D-Test-Data'
    #data_set = '1D-Solid-Boundary-Test-Data'
    # Step 1: Script Flags
    SOLVER_TYPE = {
        'Laminar Flame': False,
        'ZND Detonation': True,
        'PeleC 1D Data': False,
        'PeleC 2D Data': False,
    }

    ANALYSIS_MODE = {
        'CEMA Solver': 'Compressible',  # Simple or Compressible
        'Chemical Jacobian Solver': 'Pyjac', # Cantera or Pyjac
    }

    # Step 2: Set reactant mixture parameters
    initialize_parameters(
        T=700,
        P=ct.one_atm,
        Phi=1.0,
        Fuel='H2',
        nitrogenAmount=0.5 * 3.76,
        mech='../Chemical-Mechanisms/Li-Dryer-H2-mechanism.yaml',
        #mech='../Chemical-Mechanisms/sandiego_mechCK.yaml',
    )

    # Step 3: Create or extract data from file, depending on SOLVER_TYPE

    if sum(value is True for value in SOLVER_TYPE.values() if isinstance(value, bool)) > 1:
        raise ValueError('Please Select only one data method to be processed')

    if SOLVER_TYPE.get('Laminar Flame', False):
        tmp_data = data_handling_functions.cantera_flame(input_params, width=1e-3)
        print("Cantera Flame Simulation Completed")
        collective_data = np.array([tmp_data])

    if SOLVER_TYPE.get('ZND Detonation', False):
        tmp_data = data_handling_functions.sdtoolbox_detonation(input_params)
        print("SDToolbox ZND Simulation Completed")
        collective_data = np.array([tmp_data])

    if SOLVER_TYPE.get('PeleC 1D Data', False) or SOLVER_TYPE.get('PeleC 2D Data', False):
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
        updated_data_list = [updated_data_list[1]]

        # Step 3:
        ddt_box_size = 1e-2 * 0.01  # m
        # ddt_y_loc = 0.0462731 # 1D Data Location
        ddt_y_loc = 0.0462731 * 0.01   # 2D Data Location
        # input_params.species.remove("N2")
        # If multiple plot files are provided, determine the size of the box to fit data from the first and last file
        if len(np.array(updated_data_list)) > 1:
            initial_box = flame_box(updated_data_list[0], flame_y_loc=ddt_y_loc, box_size=ddt_box_size)
            final_box = flame_box(updated_data_list[-1], flame_y_loc=ddt_y_loc, box_size=ddt_box_size)
        else:
            initial_box = final_box = flame_box(updated_data_list[0], flame_y_loc=ddt_y_loc, box_size=ddt_box_size)

        x_bnds, y_bnds = [np.minimum(initial_box[0], final_box[0]), np.maximum(initial_box[1], final_box[1])]

        # Extract the data from the plot files in parallel
        print('Beginning Data Import')
        collective_data = []
        for data_str in updated_data_list:
            print(f'Processing: {data_str}')
            tmp_var = pelec_flame_data(data_str, x_bnds, y_bnds, ddt_y_loc, ANALYSIS_MODE)
            collective_data.append(tmp_var)
        print('Completed Data Import')

    #
    for data in collective_data:
        print('Beginning CEMA Analysis')
        if ANALYSIS_MODE['CEMA Solver'] == 'Simple':
            phi_w, phi_f, alpha = cema_functions.cema_solver(data,
                                                             input_params,
                                                             ANALYSIS_MODE['CEMA Solver'],
                                                             ANALYSIS_MODE['Chemical Jacobian Solver'])

        elif ANALYSIS_MODE['CEMA Solver'] == 'Compressible':
            phi_w, phi_d, phi_r, phi_T, phi_f, alpha = cema_functions.cema_solver(data,
                                                                                  input_params,
                                                                                  ANALYSIS_MODE['CEMA Solver'],
                                                                                  ANALYSIS_MODE['Chemical Jacobian Solver'])

        if data['Dimension'] == 1:

            plt.figure(figsize=(8, 6))
            plt.plot(data['Grid'][0, :, 0], data['Temperature'][0, :], 'k-', label='Temperature')
            plt.xlabel('Distance (cm)')
            plt.ylabel('Temperature (K)')
            plt.show()

            progress_var = data['Y(H2O)'][0, :]

            plt.figure(figsize=(8, 6))
            plt.plot(progress_var, phi_w[0, :], 'r-', label='phi_w')
            plt.plot(progress_var, phi_f[0, :], 'g-', label='phi_f')

            plt.xlabel('Distance (m)')
            plt.ylabel('Phi')  # Label for y-axis (optional)
            plt.yscale('log')  # Set y-axis to log scale
            plt.ylim(1, max(phi_w.max(), phi_f.max()))  # Set y-axis limit from 1 to max value
            plt.legend()  # Show legend
            plt.show()

            plt.figure(figsize=(8, 6))
            plt.plot(progress_var, alpha[0, :], 'k-', label='alpha')

            plt.xlabel('Distance (m)')
            plt.ylabel('Alpha')  # Label for y-axis (optional)
            plt.yscale('log')  # Set y-axis to log scale
            plt.ylim(1, max(phi_w.max(), phi_f.max()))  # Set y-axis limit from 1 to max value
            plt.legend()  # Show legend
            plt.show()

        else:

            progress_var = 0.05
            progress_var_pts = flame_contour(data, (x_bnds, y_bnds), tracking_str='Y(H2O)',
                                             tracking_val=progress_var * (
                                                     np.nanmax(data['Y(H2O)']) - np.nanmin(data['Y(H2O)'])) + np.nanmin(
                                                 data['Y(H2O)']))

            x_arr = np.unique(data['Grid'][:, :, 0])[~np.isnan(np.unique(data['Grid'][:, :, 0]))]
            y_arr = np.unique(data['Grid'][:, :, 1])[~np.isnan(np.unique(data['Grid'][:, :, 1]))]

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
            mesh = plt.pcolormesh(x_arr, y_arr, alpha, shading='auto', cmap=cmap, norm=norm)
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

            x_line = data['Grid'][len(data['Grid'][:, 0, 0]) // 2, :, 0]
            phi_w_line = phi_w[len(data['Grid'][:, 0, 0]) // 2, :]
            phi_f_line = phi_f[len(data['Grid'][:, 0, 0]) // 2, :]

            plt.figure(figsize=(8, 6))
            plt.plot(x_line, phi_w_line, 'r-', label='phi_w')
            plt.plot(x_line, phi_f_line, 'k-', label='phi_f')
            plt.legend()
            plt.show()

    return

if __name__ == '__main__':
    main()
