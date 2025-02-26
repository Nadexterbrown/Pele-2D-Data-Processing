import os, yt, multiprocessing, re, time, textwrap, cProfile, itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.neighbors import NearestNeighbors
from sdtoolbox.thermo import soundspeed_fr
import matplotlib.animation as animation
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import dask.array as da
import cantera as ct
import numpy as np

yt.set_log_level(0)

########################################################################################################################
# Global Program Setting Variables
########################################################################################################################
flame_thickness_bin_size = 21
plotting_bnds_bin = 5
n_procs = 24

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
# Function Scripts
########################################################################################################################
def worker_function(iter_var, const_list, shared_input_params, predicate, kwargs):
    """Worker function to process data and return the result."""
    global input_params
    input_params = shared_input_params
    result = predicate((iter_var, const_list, kwargs))
    return result

def parallel_processing_function(iter_arr, const_list, predicate, **kwargs):
    """Perform parallel processing using ProcessPoolExecutor to allow dynamic task execution."""
    results = []
    with ProcessPoolExecutor(max_workers=n_procs) as executor:
        # Submit tasks asynchronously
        future_to_task = {
            executor.submit(worker_function, iter_var, const_list, input_params, predicate, kwargs): iter_var
            for iter_var in iter_arr
        }

        # Collect results as they complete
        for future in as_completed(future_to_task):
            try:
                results.append(future.result())
            except Exception as exc:
                print(f"Task generated an exception: {exc}")
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
        file_list: A list of plot files to be sorted

    Returns:
        sorted_list: A list of sorted file paths
    """
    def extract_number(file_name):
        # This regex assumes the number is at the end of the file name
        match = re.search(r'plt(\d+)$', file_name)
        return int(match.group(1)) if match else float('inf')

    return sorted(file_list, key=extract_number)

def data_smoothing_function(collective_results, master_key, slave_key, bin_size=51, degree=1):
    ###########################################
    # Internal Functions
    ###########################################
    def polynomial_fit_over_array_vectorized(x, y, bin_size, degree):
        """
        Performs polynomial fitting and evaluation over the given data using vectorization.

        Args:
            x: The x-coordinates of the data points.
            y: The y-coordinates of the data points.
            bin_size: The size of the bin for polynomial fitting.
            degree: The degree of the polynomial to fit.

        Returns:
            A tuple containing the fitted x-values, fitted y-values, and the derivatives.
        """
        half_bin = bin_size // 2
        x_fit = x[half_bin: -half_bin]
        y_fit = np.array([np.polyval(np.polyfit(x[i - half_bin: i + half_bin + 1], y[:, i - half_bin: i + half_bin + 1], degree), x[i]) for i in range(half_bin, len(x) - half_bin)])
        derivatives = np.array([np.polyval(np.polyder(np.polyfit(x[i - half_bin: i + half_bin + 1], y[:, i - half_bin: i + half_bin + 1], degree)), x[i]) for i in range(half_bin, len(x) - half_bin)])
        return x_fit, y_fit, derivatives

    ###########################################
    # Main Function
    ###########################################
    """
    Optimized smoothing function using vectorized polynomial fitting.

    Args:
        collective_results: A nested dictionary containing the data.
        master_key: The key to the main level of the nested dictionary.
        slave_key: The key to the sub-level of the nested dictionary.
        bin_size: The size of the bin for polynomial fitting.
        degree: The degree of the polynomial to fit.

    Returns:
        The updated `collective_results` dictionary with smoothed data.
    """

    # Step 1: Create the smooth sub-sub-dictionary if it doesn't exist
    if 'Smooth' not in collective_results[master_key]:
        collective_results[master_key]['Smooth'] = {}

    # Step 2: Create a new sub-sub-sub dict for the slave key
    collective_results[master_key]['Smooth'][slave_key] = {}

    # Early return if the slave_key is not 'Thermodynamic State'
    x = np.array(collective_results['Time']['Value'])
    y = np.array(collective_results[master_key][slave_key])
    if slave_key != 'Thermodynamic State':
        x_fit, y_fit, _ = polynomial_fit_over_array_vectorized(x, y, bin_size, degree)
        collective_results[master_key]['Smooth'][slave_key] = y_fit
        return collective_results

    # Step 3: Handle 'Thermodynamic State' specifically
    temp_var_arr = np.array([list(v) for v in collective_results[master_key][slave_key]])
    y = temp_var_arr.T
    x_fit, y_fit, _ = polynomial_fit_over_array_vectorized(x, y, bin_size, degree)
    collective_results[master_key]['Smooth'][slave_key] = y_fit.T

    # Step 4: Update the time values
    if 'Smooth' not in collective_results['Time']:
        collective_results['Time']['Smooth'] = {}
    collective_results['Time']['Smooth']['Value'] = x_fit

    return collective_results

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

    # Create a covering grid at the maximum level present
    data = ds.covering_grid(level=max_level,
                            left_edge=[0.0, 0.0, 0.0],
                            dims=ds.domain_dimensions * [2 ** max_level, 2 ** max_level, 1])

    # Collect unique x and y coordinates from all grids at the maximum level
    x_coords = np.unique(np.concatenate([grid['boxlib', 'x'].to_value().flatten() for grid in ds.index.grids if grid.Level == max_level]))
    y_coords = np.unique(np.concatenate([grid['boxlib', 'y'].to_value().flatten() for grid in ds.index.grids if grid.Level == max_level]))

    grid_arr = np.array([x_coords, y_coords], dtype=object)

    # Determine the y-slice index and location based on the desired y-location
    if isinstance(desired_y_location, str):
        if desired_y_location == "Bottom":
            y_slice_index = 0
            y_slice_loc = data.LeftEdge[1].to_value()
        elif desired_y_location == "Top":
            y_slice_index = data.ActiveDimensions[1] - 1
            y_slice_loc = data.RightEdge[1].to_value()
        elif desired_y_location == "DDT":
            y_slice_index = np.unravel_index(np.argmax(data['boxlib', 'pressure'].to_value(), axis=None),
                                             data['boxlib', 'pressure'].to_value().shape)[1]
            y_slice_loc = data['boxlib', 'y'][0][y_slice_index].to_value()[0]
        else:
            y_slice_index = data.ActiveDimensions[1] // 2 - 1
            y_slice_loc = data['boxlib', 'y'][0][y_slice_index].to_value()[0]
    else:
        y_slice_index = np.argwhere(data["boxlib", 'y'][0][:].to_value() <= desired_y_location)[-1][0]
        y_slice_loc = data['boxlib', 'y'][0][y_slice_index].to_value()[0]

    return (np.array([[0, y_slice_index], [data.ActiveDimensions[0], y_slice_index]]),
            np.array([[data.LeftEdge[0].to_value(), y_slice_loc], [data.RightEdge[0].to_value(), y_slice_loc]]),
            grid_arr)

def plt_var_bnds(args):
    ###########################################
    # Main Function
    ###########################################
    global input_params
    iter_var, const_arr, kwargs = args

    temp_plt_files = iter_var
    domain_info = kwargs.get('domain_info', [])
    CHECK_FLAGS = kwargs.get('CHECK_FLAGS', [])

    # Step 2: Create an array of the keys where the value is True
    keys_with_true_values = [
        key for key, value in CHECK_FLAGS['Domain State Animations'].items()
        if key not in {'Combined', 'Flame Thickness', 'Surface Contour'} and (
                (isinstance(value, (tuple, list, np.ndarray)) and value[0]) or
                (isinstance(value, bool) and value) or
                any(key in combined for combined in CHECK_FLAGS['Domain State Animations'].get('Combined', []))
        )
    ]
    # Step 2: Initialize the temp_max_val_arr with the same length as the number of True values in the dictionary
    temp_bounds_arr = np.empty((len(keys_with_true_values), 3), dtype=object)

    # Step 3: Load data and collect the relevant parameters
    raw_data = yt.load(temp_plt_files)
    slice = raw_data.ray(
        np.array([domain_info[1][0][0], domain_info[1][0][1], 0.0]),
        np.array([domain_info[1][1][0], domain_info[1][1][1], 0.0])
    )

    for i, key in enumerate(keys_with_true_values):
        if key == 'Temperature':
            temp_arr = slice['boxlib', 'Temp'].to_value()
        elif key == 'Pressure':
            temp_arr = slice['boxlib', 'pressure'].to_value()
        elif key == 'Velocity':
            temp_arr = slice['boxlib', 'x_velocity'].to_value()
        elif key == 'Species':
            print('Max Value Determination for Species is W.I.P.')
            continue
        elif key == 'Heat Release Rate Cantera':
            temp_arr, _ = heat_release_rate_extractor('Cantera', plt_data=slice, sort_arr=np.argsort(slice['boxlib', 'x']))
        elif key == 'Heat Release Rate PeleC':
            try:
                temp_arr, _ = heat_release_rate_extractor('PeleC', plt_data=slice, sort_arr=np.argsort(slice['boxlib', 'x']))
            except:
                temp_arr, _ = heat_release_rate_extractor('Cantera', plt_data=slice, sort_arr=np.argsort(slice['boxlib', 'x']))
        else:
            continue

        # Step 3: Write bounds to value
        temp_bounds_arr[i, 0] = key
        temp_bounds_arr[i, 1] = np.min(temp_arr)
        temp_bounds_arr[i, 2] = np.max(temp_arr)

    return temp_bounds_arr

def animation_axis(plt_dirs, ddt_plt_file, domain_info, CHECK_FLAGS):
    ###########################################
    # Main Function
    ###########################################
    print('Begin Individual Variable Bounds')

    ddt_idx = plt_dirs.index(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Raw-PeleC-Data', ddt_plt_file))
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
    def plot_frame(reference_loc=None):
        def plot_axis(ax, x_data, y_data, label, linestyle, color, ylabel, ylim, log_scale=False):
            """Helper function to plot data on a single axis."""
            ax.plot(x_data, y_data, label=label, linestyle=linestyle, color=color)
            ax.set_ylabel(ylabel)
            ax.set_ylim(*ylim)
            if log_scale:
                ax.set_yscale('log')
            ax.legend(loc='best')

        # Step 1: Create the figure and axes
        fig, ax1 = plt.subplots()
        axes = [ax1]

        def get_filtered_data(reference_loc, x_data_arr, y_data_arr):
            if reference_loc is not None:
                indices = np.where(
                    (x_data_arr >= x_data_arr[reference_loc] - reference_loc / 2) &
                    (x_data_arr <= x_data_arr[reference_loc] + reference_loc / 2)
                )[0]
                return x_data_arr[indices], y_data_arr[indices]
            return x_data_arr, y_data_arr

        if isinstance(var_name, (np.ndarray, list, tuple)):
            for i, obj in enumerate(var_name):
                tmp_x_data, tmp_y_data = get_filtered_data(reference_loc, x_data_arr, y_data_arr)
                if i == 0:
                    plot_axis(ax=axes[0], x_data=tmp_x_data, y_data=tmp_y_data, label=obj, linestyle='-', color='k', ylabel=obj, ylim=(y_bounds[i][0], y_bounds[i][1]))
                    ax1.set_xlabel('Position [cm]')
                    ax1.grid(True, axis='x')
                else:
                    ax = ax1.twinx()
                    ax.spines["right"].set_position(("outward", 60 * i))
                    plot_axis(ax=ax, x_data=tmp_x_data, y_data=tmp_y_data, label=obj, linestyle='--', color=f"C{i}", ylabel=obj, ylim=(y_bounds[i][0], y_bounds[i][1]), log_scale=(obj == "pressure"))
                    axes.append(ax)
            title = " and ".join(var_name) + f" Variation at y = {domain_size[1][1][1]} cm and t = {time} s"
        else:
            tmp_x_data, tmp_y_data = get_filtered_data(reference_loc, x_data_arr, y_data_arr)
            plot_axis(ax=ax1, x_data=tmp_x_data, y_data=tmp_y_data, label=var_name, linestyle='-', color='k', ylabel=var_name, ylim=(y_bounds[0], y_bounds[1]), log_scale=(var_name == "Pressure"))
            title = f"{var_name} Variation at y = {domain_size[1][1][1]} cm and t = {time} s"

        wrapped_title = "\n".join(textwrap.wrap(title, width=55))
        plt.suptitle(wrapped_title, ha="center")

        ax1.set_xlim(x_bounds if x_bounds is not None else (0, domain_size[1][1][0]))
        plt.tight_layout()

        formatted_time = f"{time:.16f}".rstrip('0').rstrip('.')
        # Create filename and save plot
        if isinstance(var_name, (np.ndarray, list, tuple)):
            filename = os.path.join(output_dir_path, f"{'-'.join(var_name)}-Animation-Time-{formatted_time}.png")
        else:
            filename = os.path.join(output_dir_path, f"{var_name}-Animation-Time-{formatted_time}.png")

        plt.savefig(filename, format='png')
        plt.close()

    def create_animation():
        image_files = [os.path.join(folder_path, f) for f in sorted(os.listdir(folder_path)) if f.endswith('.png')]
        first_image = mpimg.imread(image_files[0])
        fig = plt.figure(figsize=(first_image.shape[1] / 100, first_image.shape[0] / 100), dpi=100)
        plt.axis('off')
        img_display = plt.imshow(first_image)

        def update(frame):
            img_display.set_array(mpimg.imread(image_files[frame]))
            return img_display,

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
    folder_path = kwargs.get('folder_path', "")
    output_dir_path = kwargs.get('output_dir_path', "")
    animation_filename = kwargs.get('animation_filename', "")

    if method == 'Plot':
        plot_frame(reference_loc=reference_loc)
    elif method == 'Animate':
        create_animation()
    else:
        print('Error: Did not define viable method argument (Plot or Animate)')

def wave_tracking(wave_type, **kwargs):
    plt_data = kwargs.get('plt_data')
    sort_arr = kwargs.get('sort_arr')
    pre_loaded_data = kwargs.get('pre_loaded_data')

    wave_type_to_pelec_str = {
        'Flame': {
            'PeleC': 'temp',
            'Pre Loaded': 'Temperature'
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

    def find_wave_idx(data, wave_type):
        if wave_type == 'Flame':
            return np.argwhere(data >= 2000)[-1][0]
        elif wave_type == 'Maximum Pressure':
            return np.argmax(data)
        elif wave_type == 'Leading Shock':
            return np.argmax(data >= 1.01 * data[-1])
        else:
            raise ValueError('Invalid Wave Type! Must be Flame, Maximum Pressure, or Leading Shock')

    data_str = wave_type_to_pelec_str[wave_type]['Pre Loaded'] if pre_loaded_data is not None else wave_type_to_pelec_str[wave_type]['PeleC']

    if pre_loaded_data is not None:
        x_arr = pre_loaded_data[np.argwhere(pre_loaded_data[0] == 'Position')[0][0]]
        wave_idx = find_wave_idx(pre_loaded_data[np.argwhere(pre_loaded_data[0] == data_str)[0][0]], wave_type)
    else:
        x_arr = plt_data['boxlib', 'x'][sort_arr].to_value()
        wave_idx = find_wave_idx(plt_data['boxlib', data_str][sort_arr].to_value(), wave_type)

    return wave_idx, x_arr[wave_idx]

def thermodynamic_state_extractor(wave_type, **kwargs):
    ###########################################
    # Internal Functions
    ###########################################
    def probe_array(wave_type):
        position_data = pre_loaded_data[np.argwhere(pre_loaded_data[0] == 'Position')[0][0]] if pre_loaded_data is not None else plt_data['boxlib', 'x'][sort_arr].to_value()
        wave_loc_adjustments = {
            'Flame': 1e-4,
            'Maximum Pressure': 0,
            'Pre-Shock': -1e-3,
            'Post-Shock': 1e-3
        }
        if wave_type not in wave_loc_adjustments:
            raise ValueError('Invalid Wave Type!')
        adjustment = wave_loc_adjustments[wave_type]
        probe_idx = np.argwhere(position_data >= (wave_loc + adjustment))[0][0]
        return probe_idx

    def cantera_soundspeed():
        species_comp = {input_params.species[i]: plt_data["boxlib", f"Y({input_params.species[i]})"][sort_arr][probe_idx].to_value() for i in range(len(input_params.species))}
        gas_obj = ct.Solution(input_params.mech)
        gas_obj.TPY = (
            pre_loaded_data[np.argwhere(pre_loaded_data[0] == 'Temperature')[0][0]][probe_idx] if pre_loaded_data is not None else plt_data['boxlib', 'Temp'][sort_arr][probe_idx].to_value(),
            pre_loaded_data[np.argwhere(pre_loaded_data[0] == 'Pressure')[0][0]][probe_idx] if pre_loaded_data is not None else plt_data['boxlib', 'pressure'][sort_arr][probe_idx].to_value() / 10,
            species_comp
        )
        return soundspeed_fr(gas_obj)

    ###########################################
    # Main Function
    ###########################################
    wave_loc = kwargs.get('wave_loc')
    plt_data = kwargs.get('plt_data')
    sort_arr = kwargs.get('sort_arr')
    pre_loaded_data = kwargs.get('pre_loaded_data')

    probe_idx = probe_array(wave_type)

    try:
        sound_speed = plt_data['boxlib', 'soundspeed'][sort_arr][probe_idx].to_value()
    except KeyError:
        sound_speed = cantera_soundspeed()

    if pre_loaded_data is not None:
        return (
            pre_loaded_data[np.argwhere(pre_loaded_data[0] == 'Temperature')[0][0]][probe_idx],
            pre_loaded_data[np.argwhere(pre_loaded_data[0] == 'Pressure')[0][0]][probe_idx],
            pre_loaded_data[np.argwhere(pre_loaded_data[0] == 'Density')[0][0]][probe_idx],
            sound_speed
        )
    else:
        return (
            plt_data['boxlib', 'Temp'][sort_arr][probe_idx].to_value(),
            plt_data['boxlib', 'pressure'][sort_arr][probe_idx].to_value() / 10,
            plt_data['boxlib', 'density'][sort_arr][probe_idx].to_value() * 1000,
            sound_speed
        )

def heat_release_rate_extractor(method, **kwargs):
    ###########################################
    # Internal Functions
    ###########################################
    def cantera_hrr():
        # Step 1: Initialize Cantera solution object
        temp_obj = ct.Solution(input_params.mech)
        # Step 2: Extract x_values and species names
        x_values = plt_data['boxlib', 'x'][sort_arr].to_value()
        species_names = input_params.species

        # Step 3: Initialize an empty list to store the heat release rates
        heat_release_rates = []

        # Step 4: Iterate over each x_value
        for i in range(len(x_values)):
            # Extract the species composition for the current x_value
            single_species_comp = {species: plt_data['boxlib', f'Y({species})'][sort_arr][i].to_value() for species in
                                   species_names}

            # Extract the TPY values for the current x_value
            if pre_loaded_data is not None:
                temp_obj.TPY = (
                    pre_loaded_data[np.argwhere(pre_loaded_data[0] == 'Temperature')[0][0]][i],
                    pre_loaded_data[np.argwhere(pre_loaded_data[0] == 'Pressure')[0][0]][i],
                    single_species_comp
                )
            else:
                temp_obj.TPY = (
                    plt_data['boxlib', 'Temp'][sort_arr][i].to_value(),
                    plt_data['boxlib', 'pressure'][sort_arr][i].to_value() * 10,
                    single_species_comp
                )

            # Calculate the heat release rate for the current x_value
            heat_release_rates.append(temp_obj.heat_release_rate)

        return np.array(heat_release_rates)

    ###########################################
    # Main Function
    ###########################################
    # Step 1:
    plt_data = kwargs.get('plt_data', None)
    sort_arr = kwargs.get('sort_arr', None)
    pre_loaded_data = kwargs.get('pre_loaded_data', None)

    # Step 2:
    if method == 'Cantera':
        heat_release_rate = cantera_hrr()
    elif method == 'PeleC':
        heat_release_rate = plt_data['boxlib', 'heatRelease'][sort_arr].to_value()
    else:
        print('Invalid method to determine Heat Release Rate!')

    return heat_release_rate, np.max(heat_release_rate)

def flame_geometry_function(raw_data, domain_info, output_dir, CHECK_FLAGS):
    ###########################################
    # Internal Functions
    ###########################################
    def plot_contour(raw_contour, sorted_contours, output_dir_path):
        plt.figure(figsize=(8, 6))
        plt.scatter(raw_contour[:, 0], raw_contour[:, 1], color='k', label='Raw Contour')
        for contour in sorted_contours:
            plt.plot(contour[:, 0], contour[:, 1], label='Sorted Flame Contour')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title(raw_data.current_time.to_value())
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        filename = os.path.join(output_dir_path, 'Animation-Frames', 'Surface Contour-Plt-Files', f"Flame-Length-Animation-Time-{raw_data.current_time.to_value():.16f}".rstrip('0').rstrip('.') + '.png')
        plt.savefig(filename, format='png')
        plt.close()

    def plot_flame_thickness_and_contour(raw_contour, region_grid, flame_x_idx, flame_y_idx, nearest_norm_points, output_dir_path):
        plt.figure(figsize=(8, 6))
        plt.scatter(raw_contour[:, 0], raw_contour[:, 1], color='k', label='Raw Contour')
        for segment in sorted_segments:
            plt.plot(segment[:, 0], segment[:, 1], label='Sorted Flame Contour')
        plt.scatter(region_grid[:, 0], region_grid[:, 1], marker='o', color='k', alpha=0.5, label='Region Grid Points')
        plt.scatter(grid[0][flame_x_idx], grid[1][flame_y_idx], marker='o', color='r', s=100, label=f'Flame Center: ({grid[0][flame_x_idx], grid[1][flame_y_idx]})')
        plt.scatter(nearest_norm_points[:, 0], nearest_norm_points[:, 1], marker='o', color='b', label='Points Along Normal')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.xlim(min(region_grid[:, 0]), max(region_grid[:, 0]))
        plt.ylim(min(region_grid[:, 1]), max(region_grid[:, 1]))
        plt.title(f'Flame Normal: {raw_data.current_time.to_value()}')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        filename = ensure_long_path_prefix(os.path.join(output_dir_path, 'Animation-Frames', 'Flame-Thickness-Plt-Files', f"Flame-Thickness-Animation-Time-{raw_data.current_time.to_value():.16f}".rstrip('0').rstrip('.') + '.png'))
        plt.savefig(filename, format='png')
        plt.close()

    def filter_nonphysical_points(points, x_max_threshold):
        return points[points[:, 0] <= x_max_threshold - 1e-3]

    def sort_by_nearest_neighbors(points):
        points = np.array(points)
        buffer = 0.0125 * raw_data.domain_right_edge.to_value()[1]
        valid_indices = (points[:, 1] >= raw_data.domain_left_edge.to_value()[1] + buffer) & (points[:, 1] <= raw_data.domain_right_edge.to_value()[1] - buffer)
        points = points[valid_indices]
        nbrs = NearestNeighbors(n_neighbors=len(points), algorithm='ball_tree').fit(points)
        distances, indices = nbrs.kneighbors(points)
        origin_idx = np.argmin(points[:, 1])
        order = [origin_idx]
        distance_arr = []
        segments = []
        segment_start = 0
        for i in range(1, len(points)):
            temp_idx = np.argwhere(indices[:, 0] == order[i - 1])[0][0]
            for neighbor_idx in indices[temp_idx, 1:]:
                if neighbor_idx not in order:
                    order.append(neighbor_idx)
                    break
            distance_arr.append(np.linalg.norm(points[order[i]] - points[order[i - 1]]))
            if distance_arr[-1] > 0.1 * raw_data.domain_right_edge[1].to_value():
                segments.append(points[order][segment_start:i])
                segment_start = i
        return points[order], segments, np.sum(distance_arr)

    def acquire_flame_contour(raw_data, x_boundary):
        raw_contour = raw_data.all_data().extract_isocontours("Temp", 2000)[:, :2]
        return filter_nonphysical_points(np.unique(raw_contour, axis=0), x_boundary)

    def flame_thickness(raw_data, contour_arr, center_val, grid, output_dir_path):
        def compute_contour_normal(points):
            dx = np.gradient(points[:, 0])
            dy = np.gradient(points[:, 1])
            normals = np.column_stack((dy, -dx))
            return normals / np.linalg.norm(normals, axis=1)[:, np.newaxis]

        def closest_point_from_poly(points, line):
            t_range = np.linspace(-10, 10, 10000)
            line_points = np.column_stack((grid[0][flame_x_idx] + t_range * line[0], grid[1][flame_y_idx] + t_range * line[1]))
            line_points = line_points[(line_points[:, 0] >= min(points[:, 0])) & (line_points[:, 0] <= max(points[:, 0])) & (line_points[:, 1] >= min(points[:, 1])) & (line_points[:, 1] <= max(points[:, 1]))]
            min_distance_indices = [np.argmin(np.linalg.norm(points - line_point, axis=1)) for line_point in line_points]
            return np.unique(min_distance_indices), points[np.unique(min_distance_indices)]

        normal_vect = compute_contour_normal(contour_arr)
        flame_idx = np.argmin(abs(contour_arr[:, 1] - center_val[0][1]))
        flame_norm = normal_vect[flame_idx]
        flame_x_idx = np.argmin(abs(grid[0] - contour_arr[flame_idx][0]))
        flame_y_idx = np.argmin(abs(grid[1] - contour_arr[flame_idx][1]))
        left_edge = np.array([grid[0][flame_x_idx - (flame_thickness_bin_size // 2) - 1], grid[1][flame_y_idx - (flame_thickness_bin_size // 2) - 1], 0.0])
        right_edge = np.array([grid[0][flame_x_idx + (flame_thickness_bin_size // 2)], grid[1][flame_y_idx + (flame_thickness_bin_size // 2)], 1.0])
        region = raw_data.box(left_edge, right_edge)
        region_grid = np.dstack((region['boxlib', 'x'].to_value(), region['boxlib', 'y'].to_value()))[0]
        nearest_norm_idx, nearest_norm_points = closest_point_from_poly(region_grid, flame_norm)
        region_data = region['boxlib', 'Temp'][nearest_norm_idx].to_value()
        dx = np.diff(nearest_norm_points[:, 0] / 100)
        dy = np.diff(nearest_norm_points[:, 1] / 100)
        temp_grad_x = np.nan_to_num(np.diff(region_data) / dx, nan=0)
        temp_grad_y = np.nan_to_num(np.diff(region_data) / dy, nan=0)
        try:
            flame_thickness_val = (np.max(region_data) - np.min(region_data)) / np.max(np.sqrt(temp_grad_x ** 2 + temp_grad_y ** 2))
        except ValueError:
            flame_thickness_val = 0
        if CHECK_FLAGS.get('Domain State Animations', {}).get('Flame Thickness', False):
            temp_plt_dir = ensure_long_path_prefix(os.path.join(output_dir_path, "Animation-Frames", "Flame-Thickness-Plt-Files"))
            os.makedirs(temp_plt_dir, exist_ok=True)
            plot_flame_thickness_and_contour(contour_arr, region_grid, flame_x_idx, flame_y_idx, nearest_norm_points, output_dir_path)
        return flame_thickness_val

    ###########################################
    # Main Function
    ###########################################
    # Step 1:
    grid = domain_info[-1]
    center_val = domain_info[1]

    # Step 1: Extract the flame contour and sort the points by nearest neighbors
    raw_data.force_periodicity()
    contour_verts = acquire_flame_contour(raw_data, grid[0][-1])
    sorted_points, sorted_segments, contour_length = sort_by_nearest_neighbors(contour_verts)

    if 'Domain State Animations' in CHECK_FLAGS:
        if CHECK_FLAGS['Domain State Animations'].get('Surface Contour', False):
            temp_plt_dir = ensure_long_path_prefix(
                os.path.join(output_dir, f"Animation-Frames", f"Surface Contour-Plt-Files"))

            if os.path.exists(temp_plt_dir) is False:
                os.makedirs(temp_plt_dir, exist_ok=True)

            plot_contour(contour_verts, sorted_segments, temp_plt_dir)

    # Compute requested metrics
    results = np.empty(2, dtype=object)
    if CHECK_FLAGS['Flame'].get('Surface Length', False):
        results[0] = contour_length
    if CHECK_FLAGS['Flame'].get('Flame Thickness', False):
        results[1] = flame_thickness(raw_data, sorted_points, center_val, grid, output_dir)

    return results

def single_file_processing(args):
    ###########################################
    # Internal Functions
    ###########################################
    def load_data():
        # Step 1: Load and sort the pelec plot file data
        raw_data = yt.load(pltFile_dir)
        time = raw_data.current_time.to_value()
        slice = raw_data.ray(np.array([domain_info[1][0][0], domain_info[1][0][1], 0.0]),
                             np.array([domain_info[1][1][0], domain_info[1][1][1], 0.0]))
        ray_sort = np.argsort(slice['boxlib', 'x'])

        # Step 2: Depending on the desired pre-loaded variables
        identifier = np.array(['Identifier', 'Position', 'Temperature', 'Pressure', 'Density'])
        position = slice['boxlib', 'x'][ray_sort].to_value()
        temperature = slice['boxlib', 'Temp'][ray_sort].to_value()
        pressure = slice['boxlib', 'pressure'][ray_sort].to_value() / 10
        density = slice['boxlib', 'density'][ray_sort].to_value() * 1000

        pre_loaded_data = np.array([identifier, position, temperature, pressure, density], dtype=object)

        return time, raw_data, slice, ray_sort, pre_loaded_data

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
    time, raw_data, plt_data, sort_arr, pre_loaded_data = load_data()

    # Step 2: Process the data, extracting the desired parameters for the stated wave types
    result_dict = {'Time': {'Value': time}}

    # Helper function to handle wave tracking and thermodynamic state extraction
    def process_wave(wave_type, result_key):
        result_dict[result_key] = {}
        if CHECK_FLAGS[result_key].get('Position', False):
            if result_key in ['Pre-Shock', 'Post-Shock']:
                result_dict[result_key]['Index'], result_dict[result_key]['Position'] = result_dict['Leading Shock']['Index'], result_dict['Leading Shock']['Position']
            else:
                result_dict[result_key]['Index'], result_dict[result_key]['Position'] = wave_tracking(wave_type,
                                                                                                      pre_loaded_data=pre_loaded_data)
        if CHECK_FLAGS[result_key].get('Thermodynamic State', False):
            if 'Position' not in result_dict[result_key]:
                if result_key in ['Pre-Shock', 'Post-Shock']:
                    result_dict[result_key]['Index'], result_dict[result_key]['Position'] = result_dict['Leading Shock']['Index'], result_dict['Leading Shock']['Position']
                else:
                    result_dict[result_key]['Index'], result_dict[result_key]['Position'] = wave_tracking(wave_type,
                                                                                                          pre_loaded_data=pre_loaded_data)

            result_dict[result_key]['Thermodynamic State'] = thermodynamic_state_extractor(wave_type,
                                                                                           plt_data=plt_data,
                                                                                           sort_arr=sort_arr,
                                                                                           pre_loaded_data=pre_loaded_data,
                                                                                           wave_loc=
                                                                                           result_dict[result_key][
                                                                                               'Position'])

    # Step 3: Flame Processing
    if 'Flame' in CHECK_FLAGS:
        process_wave('Flame', 'Flame')
        if CHECK_FLAGS['Flame'].get('Relative Velocity', False):
            result_dict['Flame']['Gas Velocity'] = plt_data["boxlib", "x_velocity"][sort_arr][result_dict['Flame']['Index'] + 10].to_value() / 100
        if CHECK_FLAGS['Flame'].get('Heat Release Rate Cantera', False):
            result_dict['Flame']['Heat Release Rate Cantera Array'], result_dict['Flame']['Heat Release Rate Cantera'] = heat_release_rate_extractor('Cantera', plt_data=plt_data, sort_arr=np.argsort(plt_data['boxlib', 'x']))
        if CHECK_FLAGS['Flame'].get('Heat Release Rate PeleC', False):
            try:
                result_dict['Flame']['Heat Release Rate PeleC Array'], result_dict['Flame']['Heat Release Rate PeleC'] = heat_release_rate_extractor('PeleC', plt_data=plt_data, sort_arr=np.argsort(plt_data['boxlib', 'x']))
            except:
                result_dict['Flame']['Heat Release Rate PeleC Array'], result_dict['Flame']['Heat Release Rate PeleC'] = heat_release_rate_extractor('Cantera', plt_data=plt_data, sort_arr=np.argsort(plt_data['boxlib', 'x']))
        if CHECK_FLAGS['Flame'].get('Flame Thickness', False) or CHECK_FLAGS['Flame'].get('Surface Length', False):
            result_dict['Flame']['Surface Length'], result_dict['Flame']['Flame Thickness'] = flame_geometry_function(raw_data, domain_info, output_dir, CHECK_FLAGS)

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
            if key not in ['Surface Contour', 'Flame Thickness']:
                if key == 'Combined':
                    if isinstance(value, (np.ndarray, list, tuple)):
                        bool_check = True
                        var_name = value[0]
                        pele_name = value[1]
                    else:
                        bool_check = value
                        var_name = key

                    if bool_check:
                        x_data_arr, y_data_arr, y_lim = [], [], []
                        for i in range(len(var_name)):
                            if pre_loaded_data is not None and len(pre_loaded_data) > 0 and 'Position' in pre_loaded_data[0]:
                                x_data_arr.append(pre_loaded_data[np.argwhere(pre_loaded_data[0] == 'Position')[0][0]])
                            else:
                                x_data_arr.append(plt_data['boxlib', 'x'][sort_arr].to_value())

                            if pre_loaded_data is not None and len(pre_loaded_data) > 0 and var_name in pre_loaded_data[0]:
                                y_data_arr.append(pre_loaded_data[np.argwhere(pre_loaded_data[0] == var_name)[0][0]])
                            elif var_name == 'Heat Release Rate Cantera' or var_name == 'Heat Release Rate PeleC':
                                y_data_arr.append(result_dict['Flame'][f'{var_name} Array'])
                            else:
                                y_data_arr.append(plt_data['boxlib', pele_name][sort_arr].to_value())

                            bnd_arr_index = [item[0] for item in animation_bnds].index(var_name[i])
                            y_lim.append([animation_bnds[bnd_arr_index][1], animation_bnds[bnd_arr_index][2]])

                        temp_plt_dir = ensure_long_path_prefix(os.path.join(output_dir, f"Animation-Frames", f"{'-'.join(var_name)}-Plt-Files"))
                        os.makedirs(temp_plt_dir, exist_ok=True)
                        state_animation(method='Plot',
                                        time=time,
                                        x_data_arr=x_data_arr,
                                        y_data_arr=y_data_arr,
                                        y_bounds=y_lim,
                                        domain_size=domain_info,
                                        var_name=var_name,
                                        output_dir_path=temp_plt_dir)

                else:
                    if isinstance(value, (np.ndarray, list, tuple)):
                        bool_check = value[0]
                        var_name = key
                        pele_name = value[1]
                    else:
                        bool_check = value
                        var_name = key
                        pele_name = key

                    if bool_check:
                        if pre_loaded_data is not None and len(pre_loaded_data) > 0 and 'Position' in pre_loaded_data[0]:
                            x_data_arr = pre_loaded_data[np.argwhere(pre_loaded_data[0] == 'Position')[0][0]]
                        else:
                            x_data_arr = plt_data['boxlib', 'x'][sort_arr].to_value()
                        if pre_loaded_data is not None and len(pre_loaded_data) > 0 and var_name in pre_loaded_data[0]:
                            y_data_arr = pre_loaded_data[np.argwhere(pre_loaded_data[0] == var_name)[0][0]]
                        elif var_name == 'Heat Release Rate Cantera' or var_name == 'Heat Release Rate PeleC':
                            y_data_arr = result_dict['Flame'][f'{var_name} Array']
                        else:
                            y_data_arr = plt_data['boxlib', pele_name][sort_arr].to_value()

                        bnd_arr_index = [item[0] for item in animation_bnds].index(var_name)
                        y_lim = [animation_bnds[bnd_arr_index][1], animation_bnds[bnd_arr_index][2]]

                        temp_plt_dir = ensure_long_path_prefix(os.path.join(output_dir, f"Animation-Frames", f"{var_name}-Plt-Files"))
                        os.makedirs(temp_plt_dir, exist_ok=True)
                        state_animation(method='Plot',
                                        time=time,
                                        x_data_arr=x_data_arr,
                                        y_data_arr=y_data_arr,
                                        y_bounds=y_lim,
                                        domain_size=domain_info,
                                        var_name=var_name,
                                        output_dir_path=temp_plt_dir)

    # Step 9: Local State Animation
    if 'Local State Animation' in CHECK_FLAGS:
        local_physical_window = CHECK_FLAGS['Local State Animations']['Physical Window']
        wave_idx = result_dict[f'{CHECK_FLAGS['Local State Animations']['Wave of Interest']}']['Index'] if 'Index' in CHECK_FLAGS['Local State Animations']['Wave of Interest'] else wave_tracking(plt_data, sort_arr, pre_loaded_data=pre_loaded_data)
        for key, value in CHECK_FLAGS.get('Domain State Animations', {}).items():
            if key not in ['Surface Contour', 'Flame Thickness']:
                bool_check, var_name, pele_name = (True, value[0], value[1]) if isinstance(value, (np.ndarray, list, tuple)) else (value, key, key)
                if bool_check:
                    if pre_loaded_data is not None and len(pre_loaded_data) > 0 and 'Position' in pre_loaded_data[0]:
                        x_data_arr = pre_loaded_data[np.argwhere(pre_loaded_data[0] == 'Position')[0][0]]
                    else:
                        x_data_arr = plt_data['boxlib', 'x'][sort_arr].to_value()

                    if pre_loaded_data is not None and len(pre_loaded_data) > 0 and var_name in pre_loaded_data[0]:
                        y_data_arr = pre_loaded_data[np.argwhere(pre_loaded_data[0] == var_name)[0][0]]
                    elif var_name == 'Heat Release Rate Cantera' or var_name == 'Heat Release Rate PeleC':
                        y_data_arr = result_dict['Flame'][f'{var_name} Array']
                    else:
                        y_data_arr = plt_data['boxlib', pele_name][sort_arr].to_value()

                    bnd_arr_index = [item[0] for item in animation_bnds].index(var_name)
                    x_lim = [np.searchsorted(x_data_arr / 100, wave_idx - local_physical_window, side='left'), np.searchsorted(x_data_arr / 100, wave_idx + local_physical_window, side='right')]
                    y_lim = [animation_bnds[bnd_arr_index][1], animation_bnds[bnd_arr_index][2]]
                    temp_plt_dir = ensure_long_path_prefix(os.path.join(output_dir, f"Animation-Frames", f"{'-'.join(['Local', CHECK_FLAGS['Local State Animations']['Wave of Interest'], var_name])}-Plt-Files"))
                    os.makedirs(temp_plt_dir, exist_ok=True)
                    state_animation(method='Plot', time=time, x_data_arr=x_data_arr, y_data_arr=y_data_arr, x_bounds=x_lim, y_bounds=y_lim, domain_size=domain_info, var_name=var_name, output_dir_path=temp_plt_dir)

    return result_dict

def pelec_processing(pelec_dirs, domain_info, animation_bnds, output_dir, CHECK_FLAGS):
    ###########################################
    # Internal Functions
    ###########################################
    def file_output(file_path, smoothing_check):
        # Step 1: Dynamically create the text file header depending on the assigned flags
        header_data = ["Time [s]"]

        # Step 2: Loop over processing objectives (flame, Leading shock, maximum pressure, pre-shock, post-shock)
        for key in CHECK_FLAGS.keys():
            if key in {'Flame', 'Leading Shock', 'Maximum Pressure', 'Pre-Shock', 'Post-Shock'}:
                sub_dict = CHECK_FLAGS[key]
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
            outfile.write("#" + " ".join(f"{i+1:<55.0f}" for i in range(len(header_data))) + "\n#")
            outfile.write(" ".join(f"{header:<55s}" for header in header_data) + "\n")

            time_key = 'Smooth' if smoothing_check else 'Value'
            for i in range(len(collective_results['Time'][time_key])):
                outfile.write(f" {collective_results['Time'][time_key][i]:<55e}")
                for key in {'Flame', 'Leading Shock', 'Maximum Pressure', 'Pre-Shock', 'Post-Shock'}:
                    sub_dict = CHECK_FLAGS.get(key, {})
                    for sub_key, sub_value in sub_dict.items():
                        if sub_value:
                            if sub_key == 'Thermodynamic State':
                                if smoothing_check:
                                    outfile.write(" ".join(f"{collective_results[key]['Smooth'][sub_key][i][j]:<55e}" for j in range(len(collective_results[key][time_key][sub_key][0]))))
                                else:
                                    outfile.write(" ".join(f"{collective_results[key][sub_key][i][j]:<55e}" for j in range(len(collective_results[key][sub_key][0]))))
                            else:
                                if smoothing_check:
                                    outfile.write(f" {collective_results[key]['Smooth'][sub_key][i]:<55e}")
                                else:
                                    outfile.write(f" {collective_results[key][sub_key][i]:<55e}")
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
            if CHECK_FLAGS[key].get('Smoothing', False):
                for sub_key in CHECK_FLAGS[key]:
                    if CHECK_FLAGS[key][sub_key]:
                        data_smoothing_function(collective_results, key, sub_key)
            print(f'Completed {key} Processing')

    for key in ['Flame', 'Leading Shock', 'Maximum Pressure', 'Pre-Shock', 'Post-Shock']:
        process_key(key)

    # Step 8: Write to file, if any of the sub-dictionary values except 'Domain State Animations' are true
    print('Start Output File Writing')
    write_to_file = any(any(sub_dict.values()) for key, sub_dict in CHECK_FLAGS.items() if key != 'Domain State Animations')
    smoothing_flag = any(sub_dict.get('Smoothing', False) for sub_dict in CHECK_FLAGS.values())

    if write_to_file:
        file_output(ensure_long_path_prefix(os.path.join(output_dir, 'Wave-Tracking-Results.txt')), False)
        if smoothing_flag:
            file_output(ensure_long_path_prefix(os.path.join(output_dir, 'Wave-Tracking-Smooth-Results.txt')), True)

    print('Completed Output File Writing')

    # Step 7: Create Variable Evolution
    print('Starting Animation Processing')
    def process_animations(animation_type, prefix=None):
        for key, value in CHECK_FLAGS.get(animation_type, {}).items():
            if key != 'Combined':
                bool_check = value[0] if isinstance(value, (np.ndarray, list, tuple)) else value
                if bool_check:
                    if prefix:
                        temp_plt_dir = ensure_long_path_prefix(os.path.join(output_dir, f"Animation-Frames", f"{prefix}-{key}-Plt-Files"))
                        animation_filename = ensure_long_path_prefix(os.path.join(output_dir, f"{key}-Evolution-Animation.mp4"))
                    else:
                        temp_plt_dir = ensure_long_path_prefix(os.path.join(output_dir, f"Animation-Frames", f"{key}-Plt-Files"))
                        animation_filename = ensure_long_path_prefix(os.path.join(output_dir, f"{prefix}-{key}-Evolution-Animation.mp4"))
                    state_animation(
                        method='Animate',
                        folder_path=temp_plt_dir,
                        animation_filename=animation_filename,
                    )

    process_animations('Domain State Animations', )
    # process_animations('Local State Animations', prefix=f"Local-{CHECK_FLAGS['Local State Animations']['Wave of Interest']}")

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
    row_idx = 'Center'
    ddt_plt_file = 'plt308219'

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
            'Smoothing': False
        },
        'Leading Shock': {
            'Position': True,
            'Velocity': True,
            'Smoothing': False
        },
        'Maximum Pressure': {
            'Position': False,
            'Thermodynamic State': False,
            'Smoothing': False
        },
        'Pre-Shock': {
            'Thermodynamic State': True,
            'Smoothing': False
        },
        'Post-Shock': {
            'Thermodynamic State': True,
            'Smoothing': False
        },
        'Domain State Animations': {
            'Temperature': (True, 'Temp'),
            'Pressure': (True, 'pressure'),
            'Velocity': (True, 'x_velocity'),
            'Species': False,
            'Heat Release Rate Cantera': True,
            'Heat Release Rate PeleC': (True, 'heatRelease'),
            'Surface Contour': False,
            'Flame Thickness': False,
            # 'Combined': (('Temperature', 'Pressure'), ('Temp', 'pressure'))
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
                     if os.path.isdir(os.path.join(dir_path, raw_data_folder)) and raw_data_folder.startswith('Raw-PeleC')
                     for time_step in os.listdir(os.path.join(dir_path, raw_data_folder))
                     if os.path.isdir(os.path.join(dir_path, raw_data_folder, time_step)) and time_step.startswith('plt')]

    # Step 4: Chronologically order the pltFiles and truncate the raw data list if skip loading is enabled
    updated_data_list = sort_files(time_data_dir)

    # Step 5: Determine the domain sizing parameters (size, # of cells)
    domain_info = domain_size_parameters(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Raw-PeleC-Data', ddt_plt_file) if row_idx == 'DDT' else updated_data_list[0], row_idx)

    # Step 6: Create the result directories
    os.makedirs(os.path.join(dir_path, f"Processed-Global-Results", f"y-{domain_info[1][0][1]:.3g}cm"), exist_ok=True)
    output_dir_path = os.path.join(dir_path, f"Processed-Global-Results", f"y-{domain_info[1][0][1]:.3g}cm")

    # Step 7:
    animation_axis_bnds = animation_axis(updated_data_list, ddt_plt_file, domain_info, CHECK_FLAGS)

    # Step 8:
    print('Beginning PeleC Processing')
    pelec_processing(updated_data_list, domain_info, animation_axis_bnds, output_dir_path, CHECK_FLAGS)
    print('Completed PeleC Processing')

    end_time = time.time()
    print(f"Execution time: {end_time - start_time} seconds")
    return

if __name__ == '__main__':
    main()