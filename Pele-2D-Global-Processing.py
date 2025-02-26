import os, yt, multiprocessing, re, time, textwrap, itertools
from scipy.interpolate import RegularGridInterpolator, griddata
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import cKDTree
from sdtoolbox.thermo import soundspeed_fr
import matplotlib.animation as animation
from matplotlib.tri import Triangulation
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cantera as ct
import numpy as np
import pandas as pd

yt.set_log_level(0)

########################################################################################################################
# Global Program Setting Variables
########################################################################################################################
version = 30

flame_thickness_bin_size = 11
flame_contour_temp = 2500
plotting_bnds_bin = 3
n_procs = 18


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
    input_params.update_composition()  # Update oxygen amount and composition
    input_params.load_mechanism_species()  # Load species from the mechanism file


########################################################################################################################
# Function Scripts
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


def data_smoothing_function(data_version, bin_size=51):
    ###########################################
    # Internal Functions
    ###########################################
    def load_data(file_path):
        """
        Load data from the specified file and extract headers from the second commented line.

        Parameters:
            file_path (str): Path to the data file.

        Returns:
            pd.DataFrame: Loaded data with properly extracted headers.
        """
        # Read the file, using the first line as comments and the second line as headers
        with open(file_path, 'r') as file:
            # Read the second commented line (header line)
            header_line = None
            comment_lines = 0
            for line in file:
                if line.startswith('#'):
                    comment_lines += 1
                    if comment_lines == 2:  # Second commented line
                        header_line = line.strip('#').strip()  # Remove '#' and extra spaces
                        break

        # Column width
        column_width = 55

        # Split the header line into individual headers based on the column width
        headers = [header_line[i:i + column_width].strip() for i in range(0, len(header_line), column_width)]

        # Load the data, skipping the comment lines and using the extracted headers
        data = pd.read_csv(file_path, sep=r'\s+', comment='#', header=None, names=headers)

        return data

    def data_smoothing_function(data, bin_size=51):
        """
        Smooth all columns (except the time column) using a first-order polynomial fit over a bin size.
        The smoothing will start after applying the half-bin offset to the time data.

        Args:
            data (pd.DataFrame): DataFrame where the first column is time, and the rest are data columns.
            bin_size (int): The size of the bin for polynomial fitting (must be odd).

        Returns:
            pd.DataFrame: DataFrame with smoothed data for all columns.
        """
        if bin_size % 2 == 0:
            raise ValueError("Bin size must be odd to center around a cell.")

        # Ensure 'Time [s]' is the first column
        time_column = 'Time [s]'  # Adjust this if the time column has a different name
        smoothed_data = pd.DataFrame(columns=data.columns)

        half_bin = bin_size // 2
        columns_to_smooth = [col for col in data.columns if col != time_column]

        save_time = True
        time_array = []
        for value_column in columns_to_smooth:
            smoothed_values = []
            for i in range(half_bin, len(data) - half_bin):  # Start from half_bin and end at len(data) - half_bin
                # Define the bin range
                start_idx = i - half_bin
                end_idx = i + half_bin + 1

                # Select data within the bin
                bin_data = data.iloc[start_idx:end_idx]

                temp_time = bin_data[time_column].values
                temp_data = bin_data[value_column].values

                # Remove NaN values from temp_data (value_column) only
                valid_idx = ~np.isnan(temp_data)
                temp_time = temp_time[valid_idx]
                temp_data = temp_data[valid_idx]

                # Perform the first-order polynomial fit if there are enough valid points
                if len(temp_data) >= 2:
                    x = temp_time
                    y = temp_data
                    coeffs = np.polyfit(x, y, 1)
                    poly = np.poly1d(coeffs)

                    # Get the smoothed value at the current time value
                    smoothed_values.append(poly(data[time_column].iloc[i]))
                else:
                    # If not enough valid points, append the original value
                    smoothed_values.append(data[value_column].iloc[i])

                if save_time:
                    time_array.append(data[time_column].iloc[i])

            save_time = False

            # Replace the original column with smoothed values
            smoothed_data[value_column] = smoothed_values

        # Add the time column to the smoothed data
        smoothed_data[time_column] = time_array

        return smoothed_data

    def save_smoothed_data(smoothed_data, output_file, input_file=None):
        """
        Save the smoothed data to a file while preserving column orientation,
        ensuring columns are 55 characters wide, and using the same header as the input file.

        Parameters:
            smoothed_data (pd.DataFrame): Smoothed data.
            output_file (str): Path to save the output file.
            input_file (str, optional): Path to the input file (used to get the original headers).
        """
        # Load headers from the input file if provided
        if input_file:
            with open(input_file, 'r') as file:
                # Skip non-header lines to get the second header line (commented line)
                comment_lines = 0
                header_line = None
                for line in file:
                    if line.startswith('#'):
                        comment_lines += 1
                        if comment_lines == 2:  # Second commented line
                            header_line = line.strip('#').strip()  # Remove '#' and extra spaces
                            break

            # Column width
            column_width = 55

            # Split the header line into individual headers based on the column width
            headers = [header_line[i:i + column_width].strip() for i in range(0, len(header_line), column_width)]

            # Update the column headers of the smoothed data with the extracted headers
            smoothed_data.columns = headers

        # Ensure that the data is written with exactly 55-character wide columns
        with open(output_file, 'w') as f:
            # Write the header
            f.write(' '.join([f"{col:<55}" for col in smoothed_data.columns]) + '\n')

            # Write the data
            for _, row in smoothed_data.iterrows():
                f.write(' '.join([f"{val:<55}" for val in row]) + '\n')

    ###########################################
    # Main Function
    ###########################################
    # Step 0: Define the vertical location if more than one folder exists
    vertical_location = None
    # Step 1: Define the input and output file paths
    dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            f"Processed-Global-Results-V{data_version}")
    # List subdirectories in the given path
    subdirectories = [d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]

    if len(subdirectories) == 1 and vertical_location is None:
        # Use the only folder if no domain_info is provided
        domain_info_folder = subdirectories[0]
        vertical_location = float(domain_info_folder.split('-')[-1].replace('cm', '').replace('y', ''))
    elif vertical_location is None:
        raise ValueError("Multiple folders exist in the directory. 'domain_info' must be provided.")

    # Construct the output directory path
    data_dir_path = os.path.join(
        dir_path,
        f"y-{vertical_location:.3g}cm"
    )

    # Step 2:
    raw_data = load_data(
        ensure_long_path_prefix(os.path.join(data_dir_path, f"Wave-Tracking-Results-V{data_version}.txt")))

    # Step 3: Smooth the data
    smoothed_data = data_smoothing_function(raw_data)

    # Step 4: Save the smoothed data
    output_file = ensure_long_path_prefix(
        os.path.join(data_dir_path, f"Wave-Tracking-Smoothed-Results-V{data_version}.txt"))
    save_smoothed_data(smoothed_data, output_file)

    return


def convert_units(value, type: str):
    if type == 'Position' or type == 'x' or type == 'y' or type == 'Flame Thickness' or type == 'Surface Length':
        return value / 100
    elif type == 'Temperature' or type == 'Temp':
        return value
    elif type == 'Pressure' or type == 'pressure':
        return value / 10
    elif type == 'Density' or type == 'density':
        return value * 1000
    elif type == 'Velocity' or type == 'Relative Velocity' or type == 'x_velocity' or type == 'y_velocity':
        return value / 100
    elif type == 'Heat Release Rate Cantera':
        return value
    elif type == 'Heat Release Rate PeleC':
        return value
    else:
        return value


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
    x_coords = np.arange(ds.domain_left_edge[0].to_value(), ds.domain_right_edge[0].to_value(), data.dds[0].to_value())
    y_coords = np.arange(ds.domain_left_edge[1].to_value(), ds.domain_right_edge[1].to_value(), data.dds[1].to_value())
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
            value_arr = convert_units(temp_arr, 'Temperature')
        elif key == 'Pressure':
            temp_arr = slice['boxlib', 'pressure'].to_value()
            value_arr = convert_units(temp_arr, 'Pressure')
        elif key == 'Velocity':
            temp_arr = slice['boxlib', 'x_velocity'].to_value()
            value_arr = convert_units(temp_arr, 'x_velocity')
        elif key == 'Species':
            print('Max Value Determination for Species is W.I.P.')
            continue
        elif key == 'Heat Release Rate Cantera':
            temp_arr, _ = heat_release_rate_extractor('Cantera', plt_data=slice,
                                                      sort_arr=np.argsort(slice['boxlib', 'x']))
            value_arr = convert_units(temp_arr, 'Heat Release Rate Cantera')
        elif key == 'Heat Release Rate PeleC':
            try:
                temp_arr, _ = heat_release_rate_extractor('PeleC', plt_data=slice,
                                                          sort_arr=np.argsort(slice['boxlib', 'x']))
                value_arr = convert_units(temp_arr, 'Heat Release Rate PeleC')
            except:
                temp_arr, _ = heat_release_rate_extractor('Cantera', plt_data=slice,
                                                          sort_arr=np.argsort(slice['boxlib', 'x']))
                value_arr = convert_units(temp_arr, 'Heat Release Rate Cantera')
        else:
            continue

        # Step 3: Write bounds to value
        temp_bounds_arr[i, 0] = key
        temp_bounds_arr[i, 1] = np.min(value_arr)
        temp_bounds_arr[i, 2] = np.max(value_arr)

    return temp_bounds_arr


def animation_axis(plt_dirs, ddt_dir, ddt_plt_file, domain_info, CHECK_FLAGS):
    ###########################################
    # Main Function
    ###########################################
    print('Begin Individual Variable Bounds')

    try:
        ddt_idx = plt_dirs.index(os.path.abspath(
            os.path.join(os.path.dirname(os.path.realpath(__file__)), ddt_dir, 'Raw-PeleC-Data', ddt_plt_file)))
    except:
        print('DDT Folder not in current directory, finding appropriate files.')
        temp_dir_path = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), ddt_dir))
        plt_dirs = [os.path.join(temp_dir_path, raw_data_folder, time_step)
                    for raw_data_folder in os.listdir(temp_dir_path)
                    if os.path.isdir(os.path.join(temp_dir_path, raw_data_folder)) and raw_data_folder.startswith(
                'Raw-PeleC')
                    for time_step in os.listdir(os.path.join(temp_dir_path, raw_data_folder))
                    if os.path.isdir(os.path.join(temp_dir_path, raw_data_folder, time_step)) and time_step.startswith(
                'plt')]
        ddt_idx = plt_dirs.index(os.path.abspath(
            os.path.join(os.path.dirname(os.path.realpath(__file__)), ddt_dir, 'Raw-PeleC-Data', ddt_plt_file)))

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

        final_results.append([row_texts[min_index_2nd_col], row_values_2nd_col[min_index_2nd_col],
                              row_values_3rd_col[max_index_3rd_col]])

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
                    plot_axis(ax=axes[0], x_data=tmp_x_data, y_data=tmp_y_data, label=obj, linestyle='-', color='k',
                              ylabel=obj, ylim=(y_bounds[i][0], y_bounds[i][1]))
                    ax1.set_xlabel('Position [cm]')
                    ax1.grid(True, axis='x')
                else:
                    ax = ax1.twinx()
                    ax.spines["right"].set_position(("outward", 60 * i))
                    plot_axis(ax=ax, x_data=tmp_x_data, y_data=tmp_y_data, label=obj, linestyle='--', color=f"C{i}",
                              ylabel=obj, ylim=(y_bounds[i][0], y_bounds[i][1]), log_scale=(obj == "pressure"))
                    axes.append(ax)
            title = " and ".join(var_name) + f" Variation at y = {domain_size[1][1][1]} cm and t = {time} s"
        else:
            tmp_x_data, tmp_y_data = get_filtered_data(reference_loc, x_data_arr, y_data_arr)
            plot_axis(ax=ax1, x_data=tmp_x_data, y_data=tmp_y_data, label=var_name, linestyle='-', color='k',
                      ylabel=var_name, ylim=(y_bounds[0], y_bounds[1]), log_scale=(var_name == "Pressure"))
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
            return np.argwhere(data >= 1.01 * data[-1])[-1][0]
        else:
            raise ValueError('Invalid Wave Type! Must be Flame, Maximum Pressure, or Leading Shock')

    data_str = wave_type_to_pelec_str[wave_type]['Pre Loaded'] if pre_loaded_data is not None else \
    wave_type_to_pelec_str[wave_type]['PeleC']

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
        position_data = pre_loaded_data[
            np.argwhere(pre_loaded_data[0] == 'Position')[0][0]] if pre_loaded_data is not None else \
        plt_data['boxlib', 'x'][sort_arr].to_value()
        wave_loc_adjustments = {
            'Flame': 1e-3,
            'Burned Gas': -1e-3,
            'Maximum Pressure': 0,
            'Pre-Shock': 1e-3,
            'Post-Shock': -1e-3
        }
        if wave_type not in wave_loc_adjustments:
            raise ValueError('Invalid Wave Type!')
        adjustment = wave_loc_adjustments[wave_type]
        probe_idx = np.argwhere(position_data >= (wave_loc + adjustment))[0][0]
        return probe_idx

    def cantera_soundspeed():
        species_comp = {
            input_params.species[i]: plt_data["boxlib", f"Y({input_params.species[i]})"][sort_arr][probe_idx].to_value()
            for i in range(len(input_params.species))}
        gas_obj = ct.Solution(input_params.mech)
        gas_obj.TPY = (
            pre_loaded_data[np.argwhere(pre_loaded_data[0] == 'Temperature')[0][0]][
                probe_idx] if pre_loaded_data is not None else plt_data['boxlib', 'Temp'][sort_arr][
                probe_idx].to_value(),
            pre_loaded_data[np.argwhere(pre_loaded_data[0] == 'Pressure')[0][0]][
                probe_idx] if pre_loaded_data is not None else plt_data['boxlib', 'pressure'][sort_arr][
                                                                   probe_idx].to_value() / 10,
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

    try:
        probe_idx = probe_array(wave_type)
    except Exception as e:
        print(f"Error Could not find thermodynamic location: {e}")
        probe_idx = -1

    try:
        sound_speed = plt_data['boxlib', 'soundspeed'][sort_arr][probe_idx].to_value() / 100
    except:
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
                    plt_data['boxlib', 'pressure'][sort_arr][i].to_value() / 10,
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

        # print(len(sorted_contours), sorted_contours)
        if isinstance(sorted_contours, list) and all(isinstance(c, np.ndarray) for c in sorted_contours):
            for contour in sorted_contours:
                contour = np.array(contour)
                plt.plot(contour[:, 0], contour[:, 1], label='Sorted Flame Contour')
        else:
            print(sorted_contours)
            contour = np.array(sorted_contours)
            plt.plot(contour[:, 0], contour[:, 1], label='Sorted Flame Contour')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title(raw_data.current_time.to_value())
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        filename = os.path.join(output_dir_path,
                                f"Flame-Length-Animation-Time-{raw_data.current_time.to_value():.16f}".rstrip(
                                    '0').rstrip('.') + '.png')
        plt.savefig(filename, format='png')
        plt.close()

    def plot_flame_thickness_and_contour(region_grid, region_temperature, contour_arr, normal_line, interpolator,
                                         output_dir_path):
        # Step 1:
        X, Y = np.meshgrid(np.unique(region_grid[:, 0]), np.unique(region_grid[:, 1]))
        # Step 2: Create a contour plot of the temperature field
        plt.figure(figsize=(8, 6))
        plt.scatter(region_grid[len(region_grid) // 2, 0], region_grid[len(region_grid) // 2, 1], marker='o',
                    color='r', s=100,
                    label=f'Flame Center: ({region_grid[len(region_grid) // 2, 0], region_grid[len(region_grid) // 2, 1]})')
        plt.scatter(X.flatten(), Y.flatten(), c=region_temperature.flatten(), cmap='hot')  # 'c' sets the colors
        plt.scatter(normal_line[:, 0], normal_line[:, 1], c=interpolator(normal_line).flatten(), cmap='hot')
        plt.plot(contour_arr[:, 0], contour_arr[:, 1], label='Sorted Flame Contour')
        plt.xlim(min(X.flatten()), max(X.flatten()))
        plt.ylim(min(Y.flatten()), max(Y.flatten()))
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.colorbar()

        plt.title(f'Flame Normal: {raw_data.current_time.to_value()}')
        filename = os.path.join(output_dir_path,
                                f"Flame-Thickness-Animation-Time-{raw_data.current_time.to_value():.16f}".rstrip(
                                    '0').rstrip('.') + '.png')
        plt.savefig(filename, format='png')
        plt.close()

    def sort_by_nearest_neighbors(points, domain_grid):
        buffer = 0.0075 * raw_data.domain_right_edge.to_value()[1]
        valid_indices = (points[:, 1] >= raw_data.domain_left_edge.to_value()[1] + buffer) & (
                points[:, 1] <= raw_data.domain_right_edge.to_value()[1] - buffer)
        points = points[valid_indices]

        # Use cKDTree for more efficient nearest neighbor search
        tree = cKDTree(points)
        origin_idx = np.argmin(np.lexsort((points[:, 0], points[:, 1])))
        order = [origin_idx]
        distance_arr = []
        segments = []
        segment_length = []
        segment_start = 0

        for i in range(1, len(points)):
            distances, indices = tree.query(points[order[i - 1]], k=len(points))
            for neighbor_idx in indices[1:]:  # Skip the first as it's the point itself
                if neighbor_idx not in order:
                    order.append(neighbor_idx)
                    break

            distance_arr.append(np.linalg.norm(points[order[i]] - points[order[i - 1]]))
            if distance_arr[-1] > 50 * (domain_grid[0][1] - domain_grid[0][0]):
                segments.append(points[order][segment_start:i])
                segment_length.append(np.sum(np.linalg.norm(np.diff(segments[-1], axis=0), axis=1)))
                segment_start = i

        # If no segments are appended, set segments equal to points
        if not segments:  # If segments is empty
            segments = [points]
            segment_length = [np.sum(np.linalg.norm(np.diff(segments[-1], axis=0), axis=1))]

        if len(np.concatenate(segments)) < 0.95 * len(points):
            nbrs = NearestNeighbors(n_neighbors=len(points), algorithm='ball_tree').fit(points)
            distances, indices = nbrs.kneighbors(points)
            origin_idx = np.argmin(points[:, 1])
            order = [origin_idx]
            distance_arr = []
            segments = []
            segment_length = []
            segment_start = 0
            for i in range(1, len(points)):
                temp_idx = np.argwhere(indices[:, 0] == order[i - 1])[0][0]
                for neighbor_idx in indices[temp_idx, 1:]:
                    if neighbor_idx not in order:
                        order.append(neighbor_idx)
                        break

                distance_arr.append(np.linalg.norm(points[order[i]] - points[order[i - 1]]))
                if distance_arr[-1] > 50 * (domain_grid[0][1] - domain_grid[0][0]):
                    segments.append(points[order][segment_start:i])
                    segment_length.append(np.sum(np.linalg.norm(np.diff(segments[-1], axis=0), axis=1)))
                    segment_start = i

        return points[order], segments, np.sum(segment_length)

    def manually_aquire_flame_contour(raw_data):
        # Get the maximum refinement level
        max_level = raw_data.index.max_level
        # Initialize containers for the highest level data
        x_coords, y_coords, temperatures = [], [], []
        # Loop through grids and extract data at the highest level
        for grid in raw_data.index.grids:
            if grid.Level == max_level:
                x_coords.append(grid["boxlib", "x"].to_value().flatten())
                y_coords.append(grid["boxlib", "y"].to_value().flatten())
                temperatures.append(grid["Temp"].flatten())

        x_coords = np.concatenate(x_coords)
        y_coords = np.concatenate(y_coords)
        temperatures = np.concatenate(temperatures)

        # Create a triangulation
        triangulation = Triangulation(x_coords, y_coords)

        # Use tricontour to compute the contour line
        contour = plt.tricontour(triangulation, temperatures, levels=[flame_contour_temp])

        # If no contour is found, use grid interpolation
        if not contour.collections:
            print("No contour found at the specified level. Using interpolation...")

            # Create a regular grid for interpolation
            xi = np.linspace(np.min(x_coords), np.max(x_coords), 1e4)
            yi = np.linspace(np.min(y_coords), np.max(y_coords), 1e4)
            xi, yi = np.meshgrid(xi, yi)

            # Interpolate temperatures onto the regular grid
            temperature_grid = griddata((x_coords, y_coords), temperatures, (xi, yi), method='cubic')

            # Create a new triangulation on the regular grid and compute the contour
            triangulation = Triangulation(xi.flatten(), yi.flatten())
            contour = plt.tricontour(triangulation, temperature_grid.flatten(), levels=[flame_contour_temp])

        # Extract the contour line vertices
        paths = contour.collections[0].get_paths()
        contour_points = np.vstack([path.vertices for path in paths])

        return contour_points

    def flame_thickness(contour_arr, center_val, output_dir_path):
        def extract_simulation_grid():
            # Step 1: Extract the flame location from the contour and simulation array
            flame_idx = np.argmin(abs(contour_arr[:, 1] - center_val[0][1]))
            flame_x, flame_y = contour_arr[flame_idx]

            # Step 2: Collect the max level grids
            max_level = raw_data.index.max_level
            grids = [grid for grid in raw_data.index.grids if grid.Level == max_level]

            # Step 3: Pre-allocate lists for subgrid data and filtered grids
            subgrid_x, subgrid_y, subgrid_temperatures = [], [], []
            filtered_grids = []

            # Step 4: Pre-extract the grid data once for efficiency
            grid_data = []
            for temp_grid in grids:
                x = temp_grid["boxlib", "x"].to_value().flatten()
                y = temp_grid["boxlib", "y"].to_value().flatten()
                temp = temp_grid["Temp"].flatten()
                grid_data.append((x, y, temp))

            # Step 5: Filter grids based on mean x difference
            for i, (x, y, temp) in enumerate(grid_data):
                # Calculate the mean x value for the current grid
                current_mean_x = np.mean(x)
                if i < len(grids) - 1:
                    # If the difference in mean x values is too large, skip the current grid
                    if current_mean_x > flame_x + 1e-2:
                        continue

                # If this grid is not skipped, append it to the filtered list
                filtered_grids.append(grids[i])
                # Collect the values from this grid
                subgrid_x.extend(x)
                subgrid_y.extend(y)
                subgrid_temperatures.extend(temp)

            subgrid_x = np.array(subgrid_x)
            subgrid_y = np.array(subgrid_y)
            subgrid_total_temperatures = np.array(subgrid_temperatures)

            # print(np.unique(subgrid_y).shape, np.unique(subgrid_y))
            return subgrid_x, subgrid_y, subgrid_total_temperatures

        def create_subgrid():
            # print(f"Flame X: {flame_x_arr_idx}, Flame Y: {flame_y_arr_idx}")
            # Step 1: Determine the number of indices to the left and right of the flame_x_idx
            left_x_indices = flame_x_arr_idx
            right_x_indices = len(flame_x_arr) - flame_x_arr_idx - 1
            # print(f"Left X Indices: {left_x_indices}, Right X Indices: {right_x_indices}")
            # Determine the smallest number of cells for the x indices
            x_indices = min(left_x_indices, right_x_indices)

            # Step 2: Determine the number of indices to the top and bottom of the flame_y_idx
            top_y_indices = flame_y_arr_idx
            bottom_y_indices = len(flame_y_arr) - flame_y_arr_idx - 1
            # print(f"Top Y Indices: {top_y_indices}, Bottom Y Indices: {bottom_y_indices}")
            # Determine the smallest number of cells for the y indices
            y_indices = min(top_y_indices, bottom_y_indices)

            # Step 3: Determine the subgrid bin size
            if min(x_indices, y_indices) < flame_thickness_bin_size:
                subgrid_bin_size = min(x_indices, y_indices)
            else:
                subgrid_bin_size = flame_thickness_bin_size

            # print(f"Subgrid Bin Size: {subgrid_bin_size}")
            # Step 4: Create subgrid with the appropriate number of indices on either side of flame_x_idx and flame_y_idx
            subgrid_flame_x = flame_x_arr[flame_x_arr_idx - subgrid_bin_size:flame_x_arr_idx + subgrid_bin_size + 1]
            subgrid_flame_y = flame_y_arr[flame_y_arr_idx - subgrid_bin_size:flame_y_arr_idx + subgrid_bin_size + 1]

            # Step 6: Create a grid of temperature values corresponding to the subgrid (subgrid_flame_x, subgrid_flame_y)
            subgrid_temperatures = np.full((len(subgrid_flame_y), len(subgrid_flame_x)), np.nan)

            # Step 6: Create a grid of temperature values corresponding to the subgrid (subgrid_flame_x, subgrid_flame_y)
            # Iterate over the subgrid (x, y) pairs and find the corresponding temperature from the collective data
            for i, y in enumerate(subgrid_flame_y):
                for j, x in enumerate(subgrid_flame_x):
                    # Find the index in the collective data that corresponds to the current (x, y)
                    matching_indices = np.where((subgrid_x == x) & (subgrid_y == y))

                    if len(matching_indices[0]) > 0:
                        # If a match is found, assign the temperature at the (x, y) position
                        try:
                            subgrid_temperatures[i, j] = subgrid_total_temperatures[matching_indices[0][0]]
                        except:
                            subgrid_temperatures[i, j] = np.nan
                            print(f"Error: Unable to assign temperature at ({x}, {y}), "
                                  f"Temperature set to previous value: {subgrid_temperatures[i, j]}")

            region_grid = np.dstack(np.meshgrid(subgrid_flame_x, subgrid_flame_y)).reshape(-1, 2)
            region_temperature = subgrid_temperatures.reshape(np.meshgrid(subgrid_flame_x, subgrid_flame_y)[0].shape)

            break_outer = False
            for i in range(2):
                if i == 0:
                    temp_arr = region_temperature
                else:
                    temp_arr = np.flip(region_temperature, axis=i - 1)

                for j in range(4):
                    temp_grid = np.rot90(temp_arr, k=j)

                    # Compute alignment score (difference between grid and contour points)
                    interpolator = RegularGridInterpolator((np.unique(region_grid[:, 0]), np.unique(region_grid[:, 1])),
                                                           temp_grid, bounds_error=False, fill_value=None)
                    contour_temps = interpolator(region_grid).reshape(
                        np.meshgrid(subgrid_flame_x, subgrid_flame_y)[0].shape)

                    """
                    plt.figure(figsize=(8, 6))
                    plt.imshow(interpolator(region_grid).reshape(np.meshgrid(subgrid_flame_x, subgrid_flame_y)[0].shape),
                               extent=[subgrid_flame_x.min(), subgrid_flame_x.max(), subgrid_flame_y.min(),
                                       subgrid_flame_y.max()],
                               origin='lower', cmap='hot', aspect='auto')
                    plt.colorbar(label='Temperature')
                    plt.show()
                    """

                    if np.all(contour_temps == region_temperature):
                        break_outer = True
                        break  # Break out of the inner loop

                if break_outer:
                    break  # Break out of the outer loop

            return region_grid, region_temperature, interpolator

        def calculate_contour_normal():
            # Step 1: Compute the gradient of the contour points
            dx = np.gradient(contour_arr[:, 0])
            dy = np.gradient(contour_arr[:, 1])

            # Step 2: Compute the normals
            normals = np.zeros_like(contour_arr)
            # Case 1: If the contour is aligned with the x-axis, the normal should be along the y-axis
            for i in range(len(dx)):
                if (dx[i] == 0):  # No change in x-coordinates, thus the normal is along y-axis
                    normals[i, 0] = 0  # Normal along the y-axis (positive direction)
                    normals[i, 1] = 1  # No change in y for normal direction

                # Case 2: If the contour is aligned with the y-axis, the normal should be along the x-axis
                elif (dy[i] == 0):  # No change in y-coordinates, normal should be along x-axis
                    normals[i, 0] = 1  # No change in x for normal direction
                    normals[i, 1] = 0  # Normal along the x-axis (positive direction)

                # General case: Calculate the normal by rotating the tangent 90 degrees
                else:
                    normals[i, 0] = dy[i]  # Rotate by 90 degrees
                    normals[i, 1] = -dx[i]  # Invert the x-component of the tangent

            # Step 3: Normalize the normal vectors
            normals = normals / np.linalg.norm(normals, axis=1)[:, np.newaxis]

            return normals

        def calculate_normal_vector_line(normal_vector):
            # Step 1: Determine the spacing to be used for the normal vector
            dx = np.abs(np.unique(region_grid[:, 0])[1] - np.unique(region_grid[:, 0])[0])
            dy = np.abs(np.unique(region_grid[:, 1])[1] - np.unique(region_grid[:, 1])[0])
            t_step = min(dx, dy) / np.linalg.norm(normal_vector)  # Adjust step size for resolution

            # Center point of the array
            center_point = region_grid[region_grid.shape[0] // 2]

            # Step 2: Determine the bounds for the normal vector
            t_min_x = (np.min(region_grid[:, 0]) - center_point[0]) / normal_vector[0] if normal_vector[
                                                                                              0] != 0 else -np.inf
            t_max_x = (np.max(region_grid[:, 0]) - center_point[0]) / normal_vector[0] if normal_vector[
                                                                                              0] != 0 else np.inf
            t_min_y = (np.min(region_grid[:, 1]) - center_point[1]) / normal_vector[1] if normal_vector[
                                                                                              1] != 0 else -np.inf
            t_max_y = (np.max(region_grid[:, 1]) - center_point[1]) / normal_vector[1] if normal_vector[
                                                                                              1] != 0 else np.inf

            t_start = max(min(t_min_x, t_max_x), min(t_min_y, t_max_y))
            t_end = min(max(t_min_x, t_max_x), max(t_min_y, t_max_y))

            # Step 3: Generate t_range
            t_range = np.arange(t_start, t_end, t_step / 1e2, dtype=np.float32)

            # Step 4: Generate line points along the normal vector
            x_line_points = np.array(center_point[0] + t_range * normal_vector[0], dtype=np.float32)
            y_line_points = np.array(center_point[1] + t_range * normal_vector[1], dtype=np.float32)
            line_points = np.column_stack((x_line_points, y_line_points))

            # Step 5: Filter line points to ensure they remain within bounds
            min_x, max_x = np.min(region_grid[:, 0]), np.max(region_grid[:, 0])
            min_y, max_y = np.min(region_grid[:, 1]), np.max(region_grid[:, 1])
            line_points_filtered = line_points[
                (line_points[:, 0] >= min_x) & (line_points[:, 0] <= max_x) &
                (line_points[:, 1] >= min_y) & (line_points[:, 1] <= max_y)
                ]

            return line_points_filtered

        # Step 1: Extract the flame location from the contour and simulation array
        flame_idx = np.argmin(abs(contour_arr[:, 1] - center_val[0][1]))
        flame_x, flame_y = contour_arr[flame_idx]

        # Step 2:
        subgrid_x, subgrid_y, subgrid_total_temperatures = extract_simulation_grid()
        # print(subgrid_x.shape, subgrid_y.shape, subgrid_total_temperatures.shape)

        # Step 3: Find the nearest index to the flame contour
        flame_x_idx = np.argmin(np.abs(subgrid_x - flame_x))
        flame_y_idx = np.argmin(np.abs(subgrid_y - flame_y))
        # print(flame_x_idx, flame_y_idx)

        flame_x_arr = subgrid_x[np.abs(subgrid_y - subgrid_y[flame_y_idx]) <= 1e-12]
        flame_y_arr = subgrid_y[np.abs(subgrid_x - subgrid_x[flame_x_idx]) <= 1e-12]
        # print(flame_x_arr.shape, flame_y_arr.shape)

        flame_x_arr_idx = np.argmin(np.abs(flame_x_arr - flame_x))
        flame_y_arr_idx = np.argmin(np.abs(flame_y_arr - flame_y))

        # Step 4:
        region_grid, region_temperature, interpolator = create_subgrid()
        # print(region_grid.shape, region_temperature.shape)

        # Step 5:
        contour_normals = calculate_contour_normal()
        normal_line = calculate_normal_vector_line(contour_normals[flame_idx])
        normal_distances = np.insert(np.cumsum(np.sqrt(np.sum(np.diff(normal_line, axis=0) ** 2, axis=1))), 0, 0)
        normal_line_temperature = interpolator(normal_line)

        # Step 6:
        temp_grad = np.abs(np.gradient(interpolator(normal_line)) / np.gradient(normal_distances))
        try:
            flame_thickness_val = (np.max(interpolator(normal_line)) - np.min(interpolator(normal_line))) / np.max(
                temp_grad)

            temp_plt_dir = ensure_long_path_prefix(
                os.path.join(output_dir_path, "Animation-Frames", "Flame Thickness-Plt-Files"))
            os.makedirs(temp_plt_dir, exist_ok=True)
            plot_flame_thickness_and_contour(region_grid, region_temperature, contour_arr, normal_line,
                                             interpolator, temp_plt_dir)
        except ValueError as e:
            print(f"Error: Unable to calculate flame thickness: {e}")
            flame_thickness_val = 0

        return flame_thickness_val

    ###########################################
    # Main Function
    ###########################################
    # Step 1:
    domain_grid = domain_info[-1]
    center_val = domain_info[1]

    results = np.empty(2, dtype=object)
    # Step 1: Extract the flame contour and sort the points by nearest neighbors
    raw_data.force_periodicity()
    try:
        contour_verts = manually_aquire_flame_contour(raw_data)
        sorted_points, sorted_segments, contour_length = sort_by_nearest_neighbors(contour_verts, domain_grid)

        if 'Domain State Animations' in CHECK_FLAGS:
            if CHECK_FLAGS['Domain State Animations'].get('Surface Contour', False):
                temp_plt_dir = ensure_long_path_prefix(
                    os.path.join(output_dir, f"Animation-Frames", f"Surface Contour-Plt-Files"))

                if os.path.exists(temp_plt_dir) is False:
                    os.makedirs(temp_plt_dir, exist_ok=True)

                plot_contour(contour_verts, sorted_segments, temp_plt_dir)

    except Exception as e:
        contour_length = np.nan
        print(f"Error: Unable to extract flame contour: {e}")

    # Compute requested metrics
    if CHECK_FLAGS['Flame'].get('Surface Length', False):
        results[0] = contour_length
    if CHECK_FLAGS['Flame'].get('Flame Thickness', False):
        if contour_length != 0:
            try:
                results[1] = flame_thickness(sorted_points, center_val, output_dir)
            except Exception as e:
                results[1] = np.nan
                print(f"Error: Unable to extract flame thickness: {e}")
        else:
            results[1] = np.nan

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
        pressure = slice['boxlib', 'pressure'][ray_sort].to_value()
        density = slice['boxlib', 'density'][ray_sort].to_value()

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
                result_dict[result_key]['Index'], result_dict[result_key]['Position'] = result_dict['Leading Shock'][
                    'Index'], result_dict['Leading Shock']['Position']
            elif result_key in ['Burned Gas']:
                result_dict[result_key]['Index'], result_dict[result_key]['Position'] = result_dict['Flame']['Index'], \
                result_dict['Flame']['Position']
            else:
                try:
                    result_dict[result_key]['Index'], result_dict[result_key]['Position'] = wave_tracking(wave_type,
                                                                                                          pre_loaded_data=pre_loaded_data)
                except Exception as e:
                    print(f"Error: Unable to determine {pltFile_dir} {result_key} position: {e}")
                    result_dict[result_key]['Index'] = 0
                    result_dict[result_key]['Position'] = 0

        if CHECK_FLAGS[result_key].get('Thermodynamic State', False):
            if 'Position' not in result_dict[result_key]:
                if result_key in ['Pre-Shock', 'Post-Shock']:
                    result_dict[result_key]['Index'], result_dict[result_key]['Position'] = \
                    result_dict['Leading Shock']['Index'], result_dict['Leading Shock']['Position']
                elif result_key in ['Burned Gas']:
                    result_dict[result_key]['Index'], result_dict[result_key]['Position'] = result_dict['Flame'][
                        'Index'], result_dict['Flame']['Position']
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
            result_dict['Flame']['Gas Velocity'] = plt_data["boxlib", "x_velocity"][sort_arr][
                result_dict['Flame']['Index'] + 10].to_value()
        if CHECK_FLAGS['Flame'].get('Heat Release Rate Cantera', False):
            result_dict['Flame']['Heat Release Rate Cantera Array'], result_dict['Flame'][
                'Heat Release Rate Cantera'] = heat_release_rate_extractor('Cantera', plt_data=plt_data,
                                                                           sort_arr=np.argsort(plt_data['boxlib', 'x']))
        if CHECK_FLAGS['Flame'].get('Heat Release Rate PeleC', False):
            try:
                result_dict['Flame']['Heat Release Rate PeleC Array'], result_dict['Flame'][
                    'Heat Release Rate PeleC'] = heat_release_rate_extractor('PeleC', plt_data=plt_data,
                                                                             sort_arr=np.argsort(
                                                                                 plt_data['boxlib', 'x']))
            except:
                result_dict['Flame']['Heat Release Rate PeleC Array'], result_dict['Flame'][
                    'Heat Release Rate PeleC'] = heat_release_rate_extractor('Cantera', plt_data=plt_data,
                                                                             sort_arr=np.argsort(
                                                                                 plt_data['boxlib', 'x']))
        if CHECK_FLAGS['Flame'].get('Flame Thickness', False) or CHECK_FLAGS['Flame'].get('Surface Length', False):
            result_dict['Flame']['Surface Length'], result_dict['Flame']['Flame Thickness'] = flame_geometry_function(
                raw_data, domain_info, output_dir, CHECK_FLAGS)

    if 'Burned Gas' in CHECK_FLAGS:
        process_wave('Burned Gas', 'Burned Gas')
        if CHECK_FLAGS['Burned Gas'].get('Velocity', False):
            result_dict['Burned Gas']['Velocity'] = plt_data["boxlib", "x_velocity"][sort_arr][
                result_dict['Flame']['Index'] - 10].to_value()

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
                            if pre_loaded_data is not None and len(pre_loaded_data) > 0 and 'Position' in \
                                    pre_loaded_data[0]:
                                x_data_arr.append(pre_loaded_data[np.argwhere(pre_loaded_data[0] == 'Position')[0][0]])
                            else:
                                x_data_arr.append(plt_data['boxlib', 'x'][sort_arr].to_value())

                            if pre_loaded_data is not None and len(pre_loaded_data) > 0 and var_name in pre_loaded_data[
                                0]:
                                y_data_arr.append(pre_loaded_data[np.argwhere(pre_loaded_data[0] == var_name)[0][0]])
                            elif var_name == 'Heat Release Rate Cantera' or var_name == 'Heat Release Rate PeleC':
                                y_data_arr.append(result_dict['Flame'][f'{var_name} Array'])
                            else:
                                y_data_arr.append(plt_data['boxlib', pele_name][sort_arr].to_value())

                            bnd_arr_index = [item[0] for item in animation_bnds].index(var_name[i])
                            y_lim.append([animation_bnds[bnd_arr_index][1], animation_bnds[bnd_arr_index][2]])

                        temp_plt_dir = ensure_long_path_prefix(
                            os.path.join(output_dir, f"Animation-Frames", f"{'-'.join(var_name)}-Plt-Files"))
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
                        if pre_loaded_data is not None and len(pre_loaded_data) > 0 and 'Position' in pre_loaded_data[
                            0]:
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

                        temp_plt_dir = ensure_long_path_prefix(
                            os.path.join(output_dir, f"Animation-Frames", f"{var_name}-Plt-Files"))
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
        wave_idx = result_dict[f'{CHECK_FLAGS['Local State Animations']['Wave of Interest']}']['Index'] if 'Index' in \
                                                                                                           CHECK_FLAGS[
                                                                                                               'Local State Animations'][
                                                                                                               'Wave of Interest'] else wave_tracking(
            plt_data, sort_arr, pre_loaded_data=pre_loaded_data)
        for key, value in CHECK_FLAGS.get('Domain State Animations', {}).items():
            if key not in ['Surface Contour', 'Flame Thickness']:
                bool_check, var_name, pele_name = (True, value[0], value[1]) if isinstance(value, (
                np.ndarray, list, tuple)) else (value, key, key)
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
                    x_lim = [np.searchsorted(x_data_arr / 100, wave_idx - local_physical_window, side='left'),
                             np.searchsorted(x_data_arr / 100, wave_idx + local_physical_window, side='right')]
                    y_lim = [animation_bnds[bnd_arr_index][1], animation_bnds[bnd_arr_index][2]]
                    temp_plt_dir = ensure_long_path_prefix(os.path.join(output_dir, f"Animation-Frames",
                                                                        f"{'-'.join(['Local', CHECK_FLAGS['Local State Animations']['Wave of Interest'], var_name])}-Plt-Files"))
                    os.makedirs(temp_plt_dir, exist_ok=True)
                    state_animation(method='Plot',
                                    time=time,
                                    x_data_arr=x_data_arr,
                                    y_data_arr=y_data_arr,
                                    x_bounds=x_lim,
                                    y_bounds=y_lim,
                                    domain_size=domain_info,
                                    var_name=var_name,
                                    output_dir_path=temp_plt_dir)

    return result_dict


def pelec_processing(pelec_dirs, domain_info, animation_bnds, output_dir, CHECK_FLAGS, SMOOTHING_FLAG):
    ###########################################
    # Internal Functions
    ###########################################
    def file_output(file_path, smoothing_check):
        # Step 1: Dynamically create the text file header depending on the assigned flags
        header_data = ["Time [s]"]

        # Step 2: Loop over processing objectives (flame, Leading shock, maximum pressure, pre-shock, post-shock)
        for key in CHECK_FLAGS.keys():
            if key in {'Flame', 'Burned Gas', 'Leading Shock', 'Maximum Pressure', 'Pre-Shock', 'Post-Shock'}:
                sub_dict = CHECK_FLAGS.get(key, {})
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
            outfile.write("#" + " ".join(f"{i + 1:<55.0f}" for i in range(len(header_data))) + "\n#")
            outfile.write(" ".join(f"{header:<55s}" for header in header_data) + "\n")

            # Step 3: Write Data
            time_key = 'Value'
            for i in range(len(collective_results['Time'][time_key])):
                # Write Time
                outfile.write(f" {collective_results['Time'][time_key][i]:<55e}")
                # Write Other Data
                for key in ('Flame', 'Burned Gas', 'Leading Shock', 'Maximum Pressure', 'Pre-Shock', 'Post-Shock'):
                    sub_dict = CHECK_FLAGS.get(key, {})
                    for sub_key, sub_value in sub_dict.items():
                        if sub_value:
                            if sub_key == 'Thermodynamic State':
                                types = ("Temperature", "Pressure", "Density", "Soundspeed")
                                temp_val = np.array([convert_units(value, t) for value, t in
                                                     zip(collective_results[key][sub_key][i], types)])
                                outfile.write(" ".join(f"{temp_val[j]:<55e}" for j in range(len(temp_val))))
                            else:
                                temp_val = convert_units(collective_results[key][sub_key][i], sub_key)
                                outfile.write(f" {temp_val:<55e}")
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
    collective_results = {
        master_key: {slave_key: [pelec_data[i][master_key][slave_key] for i in range(len(pelec_data))] for slave_key in
                     pelec_data[0][master_key]} for master_key in pelec_data[0]}

    # Step 3: Process each key in CHECK_FLAGS
    def process_key(key):
        if key in CHECK_FLAGS:
            print(f'Starting {key} Processing')
            if CHECK_FLAGS[key].get('Velocity', False):
                collective_results[key]['Velocity'] = np.gradient(collective_results[key]['Position']) / np.gradient(
                    collective_results['Time']['Value'])
            if CHECK_FLAGS[key].get('Relative Velocity', False):
                if 'Velocity' in collective_results[key]:
                    collective_results[key]['Relative Velocity'] = collective_results[key]['Velocity'] - \
                                                                   collective_results[key]['Gas Velocity']
                else:
                    print('ERROR: Must Enable Velocity Flag to compute the relative velocity')
            print(f'Completed {key} Processing')

    for key in ['Flame', 'Leading Shock', 'Pre-Shock', 'Post-Shock']:
        process_key(key)

    # Step 8: Write to file, if any of the sub-dictionary values except 'Domain State Animations' are true
    print('Start Output File Writing')
    write_to_file = any(
        any(sub_dict.values()) for key, sub_dict in CHECK_FLAGS.items() if key != 'Domain State Animations')

    if write_to_file:
        file_output(ensure_long_path_prefix(os.path.join(output_dir, f'Wave-Tracking-Results-V{version}.txt')), False)
    if SMOOTHING_FLAG:
        data_smoothing_function(version)
    print('Completed Output File Writing')

    # Step 7: Create Variable Evolution
    print('Starting Animation Processing')

    def process_animations(animation_type, prefix=None):
        for key, value in CHECK_FLAGS.get(animation_type, {}).items():
            if key != 'Combined':
                bool_check = value[0] if isinstance(value, (np.ndarray, list, tuple)) else value
                if bool_check:
                    if prefix:
                        temp_plt_dir = ensure_long_path_prefix(
                            os.path.join(output_dir, f"Animation-Frames", f"{prefix}-{key}-Plt-Files"))
                        animation_filename = ensure_long_path_prefix(
                            os.path.join(output_dir, f"{prefix}-{key}-Evolution-Animation.mp4"))
                    else:
                        temp_plt_dir = ensure_long_path_prefix(
                            os.path.join(output_dir, f"Animation-Frames", f"{key}-Plt-Files"))
                        animation_filename = ensure_long_path_prefix(
                            os.path.join(output_dir, f"{key}-Evolution-Animation.mp4"))
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
    row_idx = 'DDT'
    ddt_dir = '../../../Domain-Length-284cm/0.09cm-Complete-Domain/Planar-Kernel-Level-6-Part-3'
    ddt_plt_file = 'plt332200'

    SMOOTHING_FLAG = True
    CHECK_FLAGS = {
        'Flame': {
            'Position': True,
            'Velocity': True,
            'Relative Velocity': True,
            'Thermodynamic State': True,
            'Heat Release Rate Cantera': True,
            'Heat Release Rate PeleC': True,
            'Flame Thickness': True,
            'Surface Length': True
        },
        'Burned Gas': {
            'Velocity': True,
            'Thermodynamic State': True
        },
        'Leading Shock': {
            'Position': True,
            'Velocity': True
        },
        'Maximum Pressure': {
            'Position': True,
            'Thermodynamic State': True
        },
        'Pre-Shock': {
            'Thermodynamic State': True
        },
        'Post-Shock': {
            'Thermodynamic State': True
        },
        'Domain State Animations': {
            'Temperature': (True, 'Temp'),
            'Pressure': (True, 'pressure'),
            'Velocity': (True, 'x_velocity'),
            'Species': False,
            'Heat Release Rate Cantera': True,
            'Heat Release Rate PeleC': (True, 'heatRelease'),
            'Surface Contour': True,
            'Flame Thickness': True,
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
                     if
                     os.path.isdir(os.path.join(dir_path, raw_data_folder)) and raw_data_folder.startswith('Raw-PeleC')
                     for time_step in os.listdir(os.path.join(dir_path, raw_data_folder))
                     if
                     os.path.isdir(os.path.join(dir_path, raw_data_folder, time_step)) and time_step.startswith('plt')]

    # Step 4: Chronologically order the pltFiles and truncate the raw data list if skip loading is enabled
    updated_data_list = sort_files(time_data_dir)

    # Step 5: Determine the domain sizing parameters (size, # of cells)
    domain_info = domain_size_parameters(
        os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), ddt_dir, 'Raw-PeleC-Data',
                                     ddt_plt_file)) if row_idx == 'DDT' else updated_data_list[0], row_idx)

    # Step 6: Create the result directories
    os.makedirs(os.path.join(dir_path, f"Processed-Global-Results-V{version}", f"y-{domain_info[1][0][1]:.3g}cm"),
                exist_ok=True)
    output_dir_path = os.path.join(dir_path, f"Processed-Global-Results-V{version}", f"y-{domain_info[1][0][1]:.3g}cm")

    # Step 7:
    animation_axis_bnds = animation_axis(updated_data_list, ddt_dir, ddt_plt_file, domain_info, CHECK_FLAGS)

    # Step 8:
    print('Beginning PeleC Processing')
    pelec_processing(updated_data_list, domain_info, animation_axis_bnds, output_dir_path, CHECK_FLAGS, SMOOTHING_FLAG)
    print('Completed PeleC Processing')

    end_time = time.time()
    print(f"Execution time: {end_time - start_time} seconds")
    return


if __name__ == '__main__':
    main()