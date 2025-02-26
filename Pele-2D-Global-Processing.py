import os, yt, multiprocessing, re, time, textwrap, cProfile
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
plotting_bnds_bin = 3
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
    """Perform parallel processing using multiprocessing.Pool with a specified number of cores."""
    with multiprocessing.Pool(processes=n_procs, initializer=init_pool, initargs=(input_params,)) as pool:
        results = pool.starmap(worker_function, [(iter_var, const_list, input_params, predicate, kwargs) for iter_var in iter_arr])
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

    # Sort the list based on the extracted number
    sorted_list = sorted(file_list, key=extract_number)
    return sorted_list

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
        values = []
        derivative = []
        x_fit = []

        for i in range(half_bin, len(x) - half_bin):
            start_index = i - half_bin
            end_index = i + half_bin + 1

            x_bin = x[start_index:end_index]
            y_bin = y[:, start_index:end_index]

            coefficients = np.polyfit(x_bin, y_bin, degree)
            value = np.polyval(coefficients, x[i])
            values.append(value)
            x_fit.append(x[i])

            poly_derivative = np.polyder(coefficients)
            derivative_value = np.polyval(poly_derivative, x[i])
            derivative.append(derivative_value)

        return np.array(x_fit), np.array(values), np.array(derivative)
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

    # Early return if the slave key is not 'Thermodynamic State'
    if slave_key != 'Thermodynamic State':
        x = np.array(collective_results['Time']['Value'])
        y = np.array(collective_results[master_key][slave_key])
        x_fit, y_fit, _ = polynomial_fit_over_array_vectorized(x, y, bin_size, degree)
        collective_results[master_key]['Smooth'][slave_key] = y_fit
        return collective_results

    # Step 3: Handle 'Thermodynamic State' specifically
    temp_var_arr = np.array([list(v) for v in collective_results[master_key][slave_key]])
    x = np.array(collective_results['Time']['Value'])
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

    :param directory_path:
    :param desired_y_location:
    :return:
    """
    # Step 2: Load the data for physical size extraction
    ds = yt.load(directory_path)
    max_level = ds.index.max_level

    # Create sudo-grid at the maximum level present
    data = ds.covering_grid(level=max_level,
                            left_edge=[0.0, 0.0, 0.0],
                            dims=ds.domain_dimensions * [2 ** max_level, 2 ** max_level, 1],
                            # And any fields to preload (this is optional!)
                            # fields=desired_varables
                            )
    # Collect unique x and y coordinates from all grids at the maximum level
    x_coords = []
    y_coords = []

    for grid in ds.index.grids:
        if grid.Level == max_level:
            x_coords.extend(grid['boxlib', 'x'].to_value().flatten())
            y_coords.extend(grid['boxlib', 'y'].to_value().flatten())

    grid_arr = np.empty(2, dtype=object)
    grid_arr[0] = np.unique(x_coords)
    grid_arr[1] = np.unique(y_coords)

    # Step 3:
    if isinstance(desired_y_location, str) is True:
        if desired_y_location == "Bottom":
            y_slice_index = 0
            y_slice_loc = data.LeftEdge[1].to_value()
        elif desired_y_location == "Top":
            y_slice_index = data.ActiveDimensions[1] - 1
            y_slice_loc = data.RightEdge[1].to_value()
        elif desired_y_location == "DDT":
            y_slice_index = np.unravel_index(np.argmax(data['boxlib', 'pressure'].to_value(), axis=None),
                                             data['boxlib', 'pressure'].to_value().shape)
            y_slice_loc = data['boxlib', 'y'][0][y_slice_index].to_value()[0]
        else:
            y_slice_index = int((data.ActiveDimensions[1] / 2) - 1)
            y_slice_loc = data['boxlib', str('y')][0][y_slice_index].to_value()[0]
    else:
        y_slice_index = np.argwhere(data["boxlib", 'y'][0][:].to_value() <= desired_y_location)[-1][0]
        y_slice_loc = data['boxlib', 'y'][0][y_slice_index].to_value()[0]

    return (np.array([[0, int(y_slice_index)],[int(data.ActiveDimensions[0]), int(y_slice_index)]]),
            np.array([[data.LeftEdge[0].to_value(), y_slice_loc], [data.RightEdge[0].to_value(), y_slice_loc]]),
            grid_arr)

def plt_var_bnds(args):
    ###########################################
    # Main Function
    ###########################################
    # Step 1:
    global input_params
    iter_var, const_arr, kwargs = args

    temp_plt_files = iter_var
    domain_info = kwargs.get('domain_info', [])
    CHECK_FLAGS = kwargs.get('CHECK_FLAGS', [])
    # Step 2: Create an array of the keys where the value is True
    keys_with_true_values = [
        key for key, value in CHECK_FLAGS['Domain State Animations'].items()
        if (isinstance(value, (np.ndarray, list, tuple)) and value[0] is True) or value is True
    ]

    # Step 2: Initialize the temp_max_val_arr with the same length as the number of True values in the dictionary
    temp_bounds_arr = np.empty((len(keys_with_true_values), 3), dtype=object)

    # Step 3: Load data and collect the relevent parameters
    raw_data = yt.load(temp_plt_files)
    slice = raw_data.ray(np.array([domain_info[1][0][0], domain_info[1][0][1], 0.0]),
                         np.array([domain_info[1][1][0], domain_info[1][1][1], 0.0]))

    for i, key in enumerate(keys_with_true_values):
        if key == 'Temperature':
            temp_arr = slice['boxlib', 'Temp'].to_value()

        if key == 'Pressure':
            temp_arr = slice['boxlib', 'pressure'].to_value()

        if key == 'Velocity':
            temp_arr = slice['boxlib', 'x_velocity'].to_value()

        if key == 'Species':
            print('Max Value Determination for Species is W.I.P.')

        if key == 'Heat Release Rate Cantera':
            [temp_arr, _] = heat_release_rate_extractor('Cantera',
                                                   plt_data=slice,
                                                   sort_arr=np.argsort(slice['boxlib', 'x']))

        if key == 'Heat Release Rate PeleC':
            try:
                [temp_arr, _] = heat_release_rate_extractor('PeleC',
                                                            plt_data=slice,
                                                            sort_arr=np.argsort(slice['boxlib', 'x']))
            except:
                [temp_arr, _] = heat_release_rate_extractor('Cantera',
                                                            plt_data=slice,
                                                            sort_arr=np.argsort(slice['boxlib', 'x']))

        """
        if key == 'Flame Thickness':
            temp_arr = flame_geometry_function(raw_data, domain_info[1][0][1], domain_info[-1], thickness_check=True)
        """

        # Step 3: Write bounds to value
        if isinstance(temp_arr, da.Array):
            # temp_arr = temp_arr.compute()
            temp_arr = temp_arr

        temp_bounds_arr[i, 0] = key
        temp_bounds_arr[i, 1] = np.min(temp_arr)
        temp_bounds_arr[i, 2] = np.max(temp_arr)

    return temp_bounds_arr

def animation_axis(plt_dirs, ddt_plt_file, domain_info, CHECK_FLAGS):
    ###########################################
    # Main Function
    ###########################################
    # Step 1:
    print('Begin Individual Variable Bounds')
    # Step 2:
    ddt_idx = plt_dirs.index(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Raw-PeleC-Data', ddt_plt_file))
    temp_plt_files = plt_dirs[max(0, ddt_idx - plotting_bnds_bin):min(len(plt_dirs), ddt_idx + plotting_bnds_bin + 1)]

    # Step 3:
    temp_max_arr = parallel_processing_function(temp_plt_files, (), plt_var_bnds,
                                                domain_info=domain_info,
                                                CHECK_FLAGS=CHECK_FLAGS)

    # Find the number of rows and columns (assuming all sub-arrays have the same structure)
    n_rows = len(temp_max_arr[0])  # Number of rows (same for all sub-arrays)
    n_cols = len(temp_max_arr)  # Number of sub-arrays
    # Initialize an empty list to store final results
    final_results = []
    # Iterate over each row
    for i in range(n_rows):
        row_values_2nd_col = []
        row_values_3rd_col = []
        row_texts = []
        # Extract the text and values from the 2nd and 3rd columns
        for j in range(n_cols):
            text = temp_max_arr[j][i][0]  # Extract the 'text' part
            value_2nd_col = temp_max_arr[j][i][1]  # Extract the value from the 2nd column
            value_3rd_col = temp_max_arr[j][i][2]  # Extract the value from the 3rd column

            row_values_2nd_col.append(value_2nd_col)
            row_values_3rd_col.append(value_3rd_col)
            row_texts.append(text)
        # Find the index of the min value in the 2nd column and max value in the 3rd column
        min_index_2nd_col = row_values_2nd_col.index(min(row_values_2nd_col))
        max_index_3rd_col = row_values_3rd_col.index(max(row_values_3rd_col))
        # Retrieve the text, min value, and max value
        min_text = row_texts[min_index_2nd_col]
        min_value = row_values_2nd_col[min_index_2nd_col]
        max_value = row_values_3rd_col[max_index_3rd_col]
        # Append the final result as a list of [text, min value, max value]
        final_results.append([min_text, min_value, max_value])

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
            # Automatically choose the best location for the legend
            ax.legend(loc='best')

        # Step 1: Create the figure and axes
        fig, ax1 = plt.subplots()

        if isinstance(var_name, (np.ndarray, list, tuple)):
            axes = [ax1]

            for i, obj in enumerate(var_name):
                # Apply reference location filter if provided
                if reference_loc is not None:
                    indices = np.where(
                        (x_data_arr >= x_data_arr[reference_loc] - reference_loc / 2) &
                        (x_data_arr <= x_data_arr[reference_loc] + reference_loc / 2)
                    )[0]
                    tmp_x_data = x_data_arr[indices]
                    tmp_y_data = y_data_arr[indices]
                else:
                    tmp_x_data = x_data_arr
                    tmp_y_data = y_data_arr

                # Plot primary axis or create secondary axes
                if i == 0:
                    plot_axis(
                        ax=axes[0],
                        x_data=tmp_x_data,
                        y_data=tmp_y_data,
                        label=obj,
                        linestyle='-',
                        color='k',
                        ylabel=obj,
                        ylim=(y_bounds[i][0], y_bounds[i][1])
                    )
                    ax1.set_xlabel('Position [cm]')
                    ax1.grid(True, axis='x')
                else:
                    ax = ax1.twinx()
                    ax.spines["right"].set_position(("outward", 60 * i))  # Offset for clarity
                    plot_axis(
                        ax=ax,
                        x_data=tmp_x_data,
                        y_data=tmp_y_data,
                        label=obj,
                        linestyle='--',
                        color=f"C{i}",  # Automatically cycle colors
                        ylabel=obj,
                        ylim=(y_bounds[i][0], y_bounds[i][1]),
                        log_scale=(obj == "pressure")
                    )
                    axes.append(ax)

                # Title
                title = " and ".join(var_name) + f" Variation at y = {domain_size[1][1][1]} cm and t = {time} s"
        else:
            # Apply reference location filter if provided
            if reference_loc is not None:
                indices = np.where(
                    (x_data_arr >= x_data_arr[reference_loc] - reference_loc / 2) &
                    (x_data_arr <= x_data_arr[reference_loc] + reference_loc / 2)
                )[0]
                tmp_x_data = x_data_arr[indices]
                tmp_y_data = y_data_arr[indices]
            else:
                tmp_x_data = x_data_arr
                tmp_y_data = y_data_arr

            plot_axis(
                ax=ax1,
                x_data=tmp_x_data,
                y_data=tmp_y_data,
                label=var_name,
                linestyle='-',
                color='k',
                ylabel=var_name,
                ylim=(y_bounds[0], y_bounds[1]),
                log_scale=(var_name == "Pressure")
            )

            # Title
            title = f"{var_name} Variation at y = {domain_size[1][1][1]} cm and t = {time} s"

        # Wrap and set title
        wrapped_title = "\n".join(textwrap.wrap(title, width=55))
        plt.suptitle(wrapped_title, ha="center")

        # Set x-axis limits and legend
        if x_bounds is not None:
            ax1.set_xlim(x_bounds[0], x_bounds[1])
        else:
            ax1.set_xlim(0, domain_size[1][1][0])

        # Adjust layout to avoid overlapping with the legend
        plt.tight_layout()

        # Format time for filename
        formatted_time = f"{time:.16f}".rstrip('0').rstrip('.')

        # Create filename and save plot
        if isinstance(var_name, (np.ndarray, list, tuple)):
            filename = os.path.join(output_dir_path, f"{'-'.join(var_name)}-Animation-Time-{formatted_time}.png")
        else:
            filename = os.path.join(output_dir_path, f"{var_name}-Animation-Time-{formatted_time}.png")

        plt.savefig(filename, format='png')
        plt.close()  # Avoid displaying inline in notebooks

        return

    def create_animation():
        # Step 1:
        image_files = [os.path.join(folder_path, f) for f in sorted(os.listdir(folder_path)) if
                       f.endswith(f'.png')]

        # Step 2: Load the first image to determine the size
        first_image = mpimg.imread(image_files[0])

        # Step 3: Create a figure with no axes
        fig = plt.figure(figsize=(first_image.shape[1] / 100, first_image.shape[0] / 100), dpi=100)
        plt.axis('off')

        # Step 4: Placeholder for the image
        img_display = plt.imshow(first_image)

        # Step 5: Update function for animation
        def update(frame):
            img_display.set_array(mpimg.imread(image_files[frame]))
            return img_display,

        # Step 6: Create animation
        ani = animation.FuncAnimation(fig, update, frames=len(image_files), blit=True)

        # Step 7: Configure writer and save animation
        writer = animation.FFMpegWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)
        #writer = animation.writers['ffmpeg']
        #writer = writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

        ani.save(animation_filename, writer=writer)

        plt.close(fig)  # Close the figure

        return
    ###########################################
    # Main Function
    ###########################################
    # Step 0: Get values from **kwargs
    time = kwargs.get('time', 0)
    x_data_arr = kwargs.get('x_data_arr', [])
    y_data_arr = kwargs.get('x_data_arr', [])
    reference_loc = kwargs.get('reference_loc', None)  # Get reference_loc from kwargs

    x_bounds = kwargs.get('x_bounds', None)
    y_bounds = kwargs.get('y_bounds', None)
    domain_size = kwargs.get('domain_size', [])

    var_name = kwargs.get('var_name', "")

    folder_path = kwargs.get('folder_path', "")
    output_dir_path = kwargs.get('output_dir_path', "")
    animation_filename = kwargs.get('animation_filename', "")

    # Step 1: Set the plotting method (domain or local) or animation creation
    if method == 'Plot':
        if reference_loc is None:
            plot_frame()
        else:
            plot_frame(reference_loc=reference_loc)

    elif method == 'Animate':
        create_animation()
    else:
        print('Error: Did not define viable method argument (Plot or Animate)')

    return

def wave_tracking(wave_type, **kwargs):
    # Step 1: Get values from **kwargs
    plt_data = kwargs.get('plt_data', None)
    sort_arr = kwargs.get('sort_arr', None)
    pre_loaded_data = kwargs.get('pre_loaded_data', None)

    # Step 2: Determine the provided wave location
    if pre_loaded_data is not None:
        # Step 3: Load/Grab the position array
        # x_arr = pre_loaded_data[np.argwhere(pre_loaded_data[0] == 'Position')[0][0]].compute()
        x_arr = pre_loaded_data[np.argwhere(pre_loaded_data[0] == 'Position')[0][0]]

        # Step 4: Given the wave tracking criterion (flame, max pressure, leading shock) determine the location of said wave
        if wave_type == 'Flame':
            #wave_idx = np.argwhere(pre_loaded_data[np.argwhere(pre_loaded_data[0] == 'Temperature')[0][0]].compute() >= 2000)[-1][0]
            wave_idx = np.argwhere(pre_loaded_data[np.argwhere(pre_loaded_data[0] == 'Temperature')[0][0]] >= 2000)[-1][0]
        elif wave_type == 'Maximum Pressure':
            # wave_idx = np.argwhere(pre_loaded_data[np.argwhere(pre_loaded_data[0] == 'Pressure')[0][0]].compute() == np.max(
            #                        pre_loaded_data[np.argwhere(pre_loaded_data[0] == 'Pressure')[0][0]].compute()))[-1][0]
            wave_idx = np.argwhere(pre_loaded_data[np.argwhere(pre_loaded_data[0] == 'Pressure')[0][0]] == np.max(
                                   pre_loaded_data[np.argwhere(pre_loaded_data[0] == 'Pressure')[0][0]]))[-1][0]
        elif wave_type == 'Leading Shock':
            try:
                # wave_idx = np.argwhere(pre_loaded_data[np.argwhere(pre_loaded_data[0] == 'Pressure')[0][0]].compute() >= 1.01 *
                #                        pre_loaded_data[np.argwhere(pre_loaded_data[0] == 'Pressure')[0][0]].compute()[-1])[-1][0]
                wave_idx = np.argwhere(pre_loaded_data[np.argwhere(pre_loaded_data[0] == 'Pressure')[0][0]] >= 1.01 *
                                       pre_loaded_data[np.argwhere(pre_loaded_data[0] == 'Pressure')[0][0]][-1])[-1][0]
            except:
                wave_idx = 0
        else:
            print('Invalid Wave Type! Must be Flame, Maximum Pressure, or Leading Shock')

    else:
        # Step 3: Load/Grab the position array
        x_arr = plt_data['boxlib', 'x'][sort_arr].to_value()

        # Step 4: Given the wave tracking criterion (flame, max pressure, leading shock) determine the location of said wave
        if wave_type == 'Flame':
            wave_idx = np.argwhere(plt_data['boxlib', 'Temp'][sort_arr].to_value() >= 2000)[-1][0]
        elif wave_type == 'Maximum Pressure':
            wave_idx = np.argwhere(plt_data['boxlib', 'pressure'][sort_arr].to_value() == np.max(
                plt_data['boxlib', 'pressure'][sort_arr].to_value()))[-1][0]
        elif wave_type == 'Leading Shock':
            try:
                wave_idx = np.argwhere(plt_data['boxlib', 'pressure'][sort_arr].to_value() >= 1.01 *
                                       plt_data['boxlib', 'pressure'][sort_arr].to_value()[-1])[-1][0]
            except:
                wave_idx = 0
        else:
            print('Invalid Wave Type! Must be Flame, Maximum Pressure, or Leading Shock')

    return wave_idx, x_arr[wave_idx]

def thermodynamic_state_extractor(wave_type, **kwargs):
    ###########################################
    # Internal Functions
    ###########################################
    def probe_array(wave_type):
        if pre_loaded_data is not None:
            if wave_type == 'Flame':
                # probe_idx = np.argwhere(pre_loaded_data[np.argwhere(pre_loaded_data[0] == 'Position')[0][0]].compute() >= (wave_loc + 1e-4))[0][0]
                probe_idx = np.argwhere(pre_loaded_data[np.argwhere(pre_loaded_data[0] == 'Position')[0][0]] >= (wave_loc + 1e-4))[0][0]
            elif wave_type == 'Maximum Pressure':
                # probe_idx = np.argwhere(pre_loaded_data[np.argwhere(pre_loaded_data[0] == 'Position')[0][0]].compute() >= (wave_loc))[0][0]
                probe_idx = np.argwhere(pre_loaded_data[np.argwhere(pre_loaded_data[0] == 'Position')[0][0]] >= (wave_loc))[0][0]
            elif wave_type == 'Pre-Shock':
                # probe_idx = np.argwhere(pre_loaded_data[np.argwhere(pre_loaded_data[0] == 'Position')[0][0]].compute() >= (wave_loc - 1e-3))[0][0]
                probe_idx = np.argwhere(pre_loaded_data[np.argwhere(pre_loaded_data[0] == 'Position')[0][0]] >= (wave_loc - 1e-3))[0][0]
            elif wave_type == 'Post-Shock':
                # probe_idx = np.argwhere(pre_loaded_data[np.argwhere(pre_loaded_data[0] == 'Position')[0][0]].compute() >= (wave_loc + 1e-3))[0][0]
                probe_idx = np.argwhere(pre_loaded_data[np.argwhere(pre_loaded_data[0] == 'Position')[0][0]] >= (wave_loc + 1e-3))[0][0]
            else:
                print('Invalid Wave Type!')
        else:
            if wave_type == 'Flame':
                probe_idx = np.argwhere(plt_data['boxlib', 'x'][sort_arr].to_value() >= (wave_loc + 1e-4))[0][0]
            elif wave_type == 'Maximum Pressure':
                probe_idx = np.argwhere(plt_data['boxlib', 'x'][sort_arr].to_value() >= (wave_loc))[0][0]
            elif wave_type == 'Pre-Shock':
                probe_idx = np.argwhere(plt_data['boxlib', 'x'][sort_arr].to_value() >= (wave_loc - 1e-3))[0][0]
            elif wave_type == 'Post-Shock':
                probe_idx = np.argwhere(plt_data['boxlib', 'x'][sort_arr].to_value() >= (wave_loc + 1e-3))[0][0]
            else:
                print('Invalid Wave Type!')

        return probe_idx

    def cantera_soundspeed():
        # Step 1: Extract species mass fractions from the pelec data
        species_comp = {}
        for i in range(len(input_params.species)):
            print(input_params.species[i])
            species_comp.update({f"{input_params.species[i]}":
                                     plt_data["boxlib", f"Y({input_params.species[i]})"][sort_arr][probe_idx].to_value()})

        # Step 2: Create a Cantera object to extract soundspeed
        gas_obj = ct.Solution(input_params.mech)
        if pre_loaded_data is not None:
            # gas_obj.TPY = (pre_loaded_data[np.argwhere(pre_loaded_data[0] == 'Temperature')[0][0]].compute()[probe_idx],
            #                pre_loaded_data[np.argwhere(pre_loaded_data[0] == 'Pressure')[0][0]].compute()[probe_idx],
            #                species_comp)
            gas_obj.TPY = (pre_loaded_data[np.argwhere(pre_loaded_data[0] == 'Temperature')[0][0]][probe_idx],
                           pre_loaded_data[np.argwhere(pre_loaded_data[0] == 'Pressure')[0][0]][probe_idx],
                           species_comp)
        else:
            gas_obj.TPY = (plt_data['boxlib', 'Temp'][sort_arr][probe_idx].to_value(),
                           plt_data['boxlib', 'pressure'][sort_arr][probe_idx].to_value() / 10,
                           species_comp)

        # Step 3: Return the soundspeed
        return soundspeed_fr(gas_obj)

    ###########################################
    # Main Function
    ###########################################
    # Step 1: Get values from **kwargs
    wave_loc = kwargs.get('wave_loc', None)
    plt_data = kwargs.get('plt_data', None)
    sort_arr = kwargs.get('sort_arr', None)
    pre_loaded_data = kwargs.get('pre_loaded_data', None)

    # Step 1:
    probe_idx = probe_array(wave_type)

    # Step 2: Calculate the sound_speed of the mixture
    try:
        sound_speed = plt_data['boxlib', 'soundspeed'][sort_arr][probe_idx].to_value()
    except:
        print(probe_idx)
        sound_speed = cantera_soundspeed()

    # Step 3:
    if pre_loaded_data is not None:
        """
        return (pre_loaded_data[np.argwhere(pre_loaded_data[0] == 'Temperature')[0][0]].compute()[probe_idx],
                pre_loaded_data[np.argwhere(pre_loaded_data[0] == 'Pressure')[0][0]].compute()[probe_idx],
                pre_loaded_data[np.argwhere(pre_loaded_data[0] == 'Density')[0][0]].compute()[probe_idx],
                sound_speed)
        """
        return (pre_loaded_data[np.argwhere(pre_loaded_data[0] == 'Temperature')[0][0]][probe_idx],
                pre_loaded_data[np.argwhere(pre_loaded_data[0] == 'Pressure')[0][0]][probe_idx],
                pre_loaded_data[np.argwhere(pre_loaded_data[0] == 'Density')[0][0]][probe_idx],
                sound_speed)

    else:
        return (plt_data['boxlib', 'Temp'][sort_arr][probe_idx].to_value(),
                plt_data['boxlib', 'pressure'][sort_arr][probe_idx].to_value() / 10,
                plt_data['boxlib', 'density'][sort_arr][probe_idx].to_value() * 1000,
                sound_speed)

def heat_release_rate_extractor(method, **kwargs):
    ###########################################
    # Internal Functions
    ###########################################
    def cantera_hrr():
        # Step 1:
        temp_obj = ct.Solution(input_params.mech)
        # Step 2:
        heat_release_rate = np.zeros(len(plt_data['boxlib', 'x'][sort_arr].to_value()))
        for i in range(len(plt_data['boxlib', 'x'][sort_arr].to_value())):
            # Step 2.1:
            species_comp = {}
            for j in range(len(input_params.species)):
                species_comp.update({f'{input_params.species[j]}':
                                         plt_data['boxlib', f'Y({input_params.species[j]})'][sort_arr][i].to_value()})
            # Step 2.2:
            if pre_loaded_data is not None:
                # temp_obj.TPY = (pre_loaded_data[np.argwhere(pre_loaded_data[0] == 'Temperature')[0][0]].compute()[i],
                #                 pre_loaded_data[np.argwhere(pre_loaded_data[0] == 'Pressure')[0][0]].compute()[i],
                #                 species_comp)

                temp_obj.TPY = (pre_loaded_data[np.argwhere(pre_loaded_data[0] == 'Temperature')[0][0]][i],
                                pre_loaded_data[np.argwhere(pre_loaded_data[0] == 'Pressure')[0][0]][i],
                                species_comp)
            else:
                temp_obj.TPY = (plt_data['boxlib', 'Temp'][sort_arr][i].to_value(),
                                plt_data['boxlib', 'pressure'][sort_arr][i].to_value() * 10,
                                species_comp)

            heat_release_rate[i] = temp_obj.heat_release_rate

        return heat_release_rate

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

    return da.from_array(heat_release_rate, chunks=(1000, )), np.max(heat_release_rate)

def flame_geometry_function(raw_data, domain_info, output_dir, CHECK_FLAGS):
    ###########################################
    # Internal Functions
    ###########################################
    def plot_contour(raw_contour, sorted_contours, output_dir_path):
        # Create the figure
        plt.figure(figsize=(8, 6))

        # Plot the flame contour
        plt.scatter(
            raw_contour[:, 0], raw_contour[:, 1], color='k', label='Raw Contour'
        )

        for i in range(len(sorted_contours)):
            plt.plot(
                sorted_contours[i][:, 0], sorted_contours[i][:, 1], label='Sorted Flame Contour'
            )

        # Configure plot labels, limits, and grid
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title(raw_data.current_time.to_value())
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()

        formatted_time = '{:.16f}'.format(raw_data.current_time.to_value()).rstrip('0').rstrip('.')
        filename = os.path.join(output_dir_path, 'Animation-Frames', 'Surface Contour-Plt-Files', f"Flame-Length-Animation-Time-{formatted_time}.png")
        plt.savefig(filename, format='png')

        # Display the plot
        plt.show()
        plt.close()

    def plot_flame_thickness_and_contour(
            raw_contour, region_grid, flame_x_idx, flame_y_idx, nearest_norm_points, output_dir_path
    ):
        """
        Plot the flame thickness and contour with key visual elements.

        Parameters:
            contour_arr (ndarray): Contour points of the flame (Nx2 array).
            region_grid (ndarray): Grid points in the region (Mx2 array).
            flame_x_idx (int): Index of the flame center in the x-dimension.
            flame_y_idx (int): Index of the flame center in the y-dimension.
            nearest_norm_points (ndarray): Points along the flame normal (Px2 array).
            flame_norm (ndarray): Normal vector at the flame center (1x2 array).
            grid (list): Grid arrays for x and y coordinates.
            raw_data (yt dataset): Dataset containing simulation data and time information.
            output_dir_path (str): Path to save the generated plot.
        """
        # Create the figure
        plt.figure(figsize=(8, 6))

        # Plot the flame contour
        plt.scatter(
            raw_contour[:, 0], raw_contour[:, 1], color='k', label='Raw Contour'
        )

        for i in range(len(sorted_segments)):
            plt.plot(
                sorted_segments[i][:, 0], sorted_segments[i][:, 1], label='Sorted Flame Contour'
            )

        # Scatter grid points in the region
        plt.scatter(
            region_grid[:, 0], region_grid[:, 1],
            marker='o', color='k', alpha=0.5, label='Region Grid Points'
        )

        # Highlight the flame center point
        plt.scatter(
            grid[0][flame_x_idx], grid[1][flame_y_idx],
            marker='o', color='r', s=100, label=f'Flame Center: ({grid[0][flame_x_idx], grid[1][flame_y_idx]})'
        )

        # Scatter points along the flame normal
        plt.scatter(
            nearest_norm_points[:, 0], nearest_norm_points[:, 1],
            marker='o', color='b', label='Points Along Normal'
        )

        # Add a quiver to show the normal direction
        """
        plt.quiver(
            grid[0][flame_x_idx], grid[1][flame_y_idx],
            flame_norm[0], flame_norm[1],
            angles='xy', scale_units='xy', scale=5, color='r', label='Normal Vector'
        )
        """

        # Configure plot labels, limits, and grid
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.xlim(min(region_grid[:, 0]), max(region_grid[:, 0]))
        plt.ylim(min(region_grid[:, 1]), max(region_grid[:, 1]))
        plt.title(f'Flame Normal: {raw_data.current_time.to_value()}')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()

        formatted_time = '{:.16f}'.format(raw_data.current_time.to_value()).rstrip('0').rstrip('.')
        filename = ensure_long_path_prefix(os.path.join(output_dir_path, 'Animation-Frames', 'Flame-Thickness-Plt-Files', f"Flame-Thickness-Animation-Time-{formatted_time}.png"))
        plt.savefig(filename, format='png')
        plt.close()

    def filter_nonphysical_points(points, x_max_threshold):
        """Filter out points near the rightmost x boundary."""
        return points[np.argwhere(points[:, 0] <= x_max_threshold - 1e-3).flatten()]

    def sort_by_nearest_neighbors(points):
        """Robustly sort points by nearest neighbors and handle disconnected or skipped points."""
        # Convert points to numpy array if it's a list of lists
        points = np.array(points)

        # Filter points within the specified y-bounds
        buffer = 0.0125 * raw_data.domain_right_edge.to_value()[1]
        valid_indices = (points[:, 1] >= raw_data.domain_left_edge.to_value()[1] + buffer) & (points[:, 1] <= raw_data.domain_right_edge.to_value()[1] - buffer)
        points = points[valid_indices]

        # Create NearestNeighbors instance
        nbrs = NearestNeighbors(n_neighbors=len(points), algorithm='ball_tree').fit(points)
        distances, indices = nbrs.kneighbors(points)

        # Construct the sorted order
        # distance_metric = lambda x: (x[0] - 0) ** 2 + (x[1] - 0) ** 2
        # origin_point = min(points, key=distance_metric)
        # origin_idx = np.argwhere(points == origin_point)[0][0]
        origin_idx = np.argmin(points[:, 1])

        # Create viable points array
        order = [origin_idx]
        neighbor_distance = []
        distance_arr = []
        segments = []
        segment_length = []

        segment_start = 0
        for i in range(1, len(points)):
            neighbors = []
            temp_idx = np.argwhere(indices[:, 0] == order[i - 1])[0][0]
            for neighbor_idx in indices[temp_idx, 1::]:
                if neighbor_idx not in order and neighbor_idx != indices[temp_idx, 0]:
                    neighbors.append(neighbor_idx)
                    # Mark the neighbor as visited
                    order.append(neighbor_idx)
                    neighbor_distance.append(distances[temp_idx, np.argwhere(indices[temp_idx, :] == neighbor_idx)])
                    # Stop if we've found enough neighbors for the given point
                    break

            distance_arr.append(np.sqrt((points[order[i], 0] - points[order[i - 1], 0]) ** 2 +
                                (points[order[i], 1] - points[order[i - 1], 1]) ** 2))
            if distance_arr[-1] > 0.1 * raw_data.domain_right_edge[1].to_value():
                segments.append(np.array(points[order][segment_start:i]))
                segment_length.append(np.sum(distance_arr[segment_start:i]))
                segment_start = i

        return points[order], segments, np.sum(segment_length)

    def acquire_flame_contour(raw_data, x_boundary):
        raw_contour = raw_data.all_data().extract_isocontours("Temp", 2000)
        contour_2d = raw_contour[:, :2]  # Retain only x and y
        contour_2d = np.unique(contour_2d, axis=0)

        return filter_nonphysical_points(contour_2d, x_boundary)

    def flame_thickness(raw_data, contour_arr, center_val, grid, output_dir_path):
        ###########################################
        # Internal Functions
        ###########################################
        def compute_contour_normal(points):
            # Create a placeholder array for the tangent
            dx = np.gradient(points[:, 0])
            dy = np.gradient(points[:, 1])
            # Compute normal vectors
            normals = np.zeros_like(points)
            normals[:, 0] = dy  # dy
            normals[:, 1] = -dx  # dx

            normal_vect = normals / np.linalg.norm(normals)
            return normal_vect

        def closest_point_from_poly(points, line):

            # Step 1: Generate points along the line defined by the unit vector
            # Define a range for the line, e.g., from -10 to 10 in the direction of the unit vector
            t_range = np.linspace(-10, 10, 10000)
            line_points  = np.column_stack((grid[0][flame_x_idx] + t_range * line[0], grid[1][flame_y_idx] + t_range * line[1]))

            within_bounds_mask = (
                    (line_points[:, 0] >= min(points[:, 0])) & (line_points[:, 0] <= max(points[:, 0])) &
                    (line_points[:, 1] >= min(points[:, 1])) & (line_points[:, 1] <= max(points[:, 1]))
            )
            line_points = line_points[within_bounds_mask]

            # Step 3: Determine the nearest point to the line generated by the unit vector
            min_distance_indices = []

            for line_point in line_points:
                line_x, line_y = line_point
                distances = []

                for point in points:
                    x, y = point
                    # Calculate perpendicular distance to the line point
                    distance = np.sqrt((line_x - x) ** 2 + (line_y - y) ** 2)
                    distances.append(distance)

                # Find the index of the minimum distance for this point
                min_distance_index = np.argmin(distances)
                min_distance_indices.append(min_distance_index)

            min_distance_indices = np.unique(min_distance_indices)
            closest_points = [points[idx] for idx in min_distance_indices]

            return min_distance_indices, np.array(closest_points)

        ###########################################
        # Main Function
        ###########################################
        """Calculate flame thickness based on temperature gradients along the flame surface normal."""
        # Step 1:
        normal_vect = compute_contour_normal(contour_arr)

        # Step 2:
        flame_idx = np.argmin(abs(contour_arr[:, 1] - center_val[0][1]))
        flame_norm = normal_vect[flame_idx]

        # Find the location of the flame
        flame_x_idx = np.argmin(abs(grid[0] - contour_arr[flame_idx][0]))
        flame_y_idx = np.argmin(abs(grid[1] - contour_arr[flame_idx][1]))

        # Create a sudo-grid region around the center flame point
        left_edge = np.array([grid[0][flame_x_idx - (flame_thickness_bin_size // 2) - 1],
                              grid[1][flame_y_idx - (flame_thickness_bin_size // 2) - 1], 0.0])

        right_edge = np.array([grid[0][flame_x_idx + (flame_thickness_bin_size // 2)],
                               grid[1][flame_y_idx + (flame_thickness_bin_size // 2)], 1.0])

        region = raw_data.box(left_edge, right_edge)

        region_grid = np.dstack((region['boxlib', 'x'].to_value(), region['boxlib', 'y'].to_value()))[0]
        # Determine the closes points in the region grid to the line created by the norm
        nearest_norm_idx, nearest_norm_points = closest_point_from_poly(region_grid, flame_norm)

        # Collect the temperature points nearest the flame surface norm
        region_data = region['boxlib', 'Temp'][nearest_norm_idx].to_value()

        # Step 2: Compute the gradient of temperature with respect to position
        # temperature_gradient = abs(np.gradient(region_data) / np.gradient(np.sqrt(temp_grad_x[len()]**2 + temp_grad_y[]**2)))
        # temperature_gradient_idx = np.argwhere(temperature_gradient == np.max(temperature_gradient))[0][0]

        # Step 3:
        dx = np.diff(nearest_norm_points[:, 0] / 100)  # cm to m
        dy = np.diff(nearest_norm_points[:, 1] / 100)  # cm to m

        dx[dx == 0] = np.nan
        dy[dy == 0] = np.nan

        temp_grad_x = np.diff(region_data) / dx
        temp_grad_y = np.diff(region_data) / dy

        temp_grad_x = np.nan_to_num(temp_grad_x, nan=0)
        temp_grad_y = np.nan_to_num(temp_grad_y, nan=0)

        try:
            flame_thickness_val = (np.max(region_data) - np.min(region_data)) / np.max(np.sqrt(temp_grad_x ** 2 + temp_grad_y ** 2))
        except:
            flame_thickness_val = 0

        if 'Domain State Animations' in CHECK_FLAGS:
            if CHECK_FLAGS['Domain State Animations'].get('Flame Thickness', False):
                temp_plt_dir = ensure_long_path_prefix(
                    os.path.join(output_dir_path, f"Animation-Frames", f"Flame-Thickness-Plt-Files"))

                if os.path.exists(temp_plt_dir) is False:
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
        """
        position = da.from_array(slice['boxlib', 'x'][ray_sort].to_value(), chunks=(1000, ))  # 1D position
        temperature = da.from_array(slice['boxlib', 'Temp'][ray_sort].to_value(), chunks=(1000, ))  # Temperature
        pressure = da.from_array(slice['boxlib', 'pressure'][ray_sort].to_value() / 10, chunks=(1000, ))  # Pressure
        density = da.from_array(slice['boxlib', 'density'][ray_sort].to_value() * 1000, chunks=(1000, ))
        """
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
    result_dict = {}
    result_dict['Time'] = {}
    result_dict['Time']['Value'] = time

    # Step 3: Flame Processing
    if 'Flame' in CHECK_FLAGS:
        # Step 3.1:
        result_dict['Flame'] = {}

        # Position
        if CHECK_FLAGS['Flame'].get('Position', False):
            [result_dict['Flame']['Index'], result_dict['Flame']['Position']] = wave_tracking('Flame', pre_loaded_data=pre_loaded_data)

        # Gas Velocity
        if CHECK_FLAGS['Flame'].get('Relative Velocity', False):
            result_dict['Flame']['Gas Velocity'] = plt_data["boxlib", "x_velocity"][sort_arr][result_dict['Flame']['Index'] + 10].to_value() / 100

        # Thermodynamic State
        if CHECK_FLAGS['Flame'].get('Thermodynamic State', False):
            result_dict['Flame']['Thermodynamic State'] = thermodynamic_state_extractor('Flame',
                                                                                        plt_data=plt_data,
                                                                                        sort_arr=sort_arr,
                                                                                        pre_loaded_data=pre_loaded_data,
                                                                                        wave_loc=result_dict['Flame']['Position'])

        # Heat Release Rate - Cantera
        if CHECK_FLAGS['Flame'].get('Heat Release Rate Cantera', False):
            [result_dict['Flame']['Heat Release Rate Cantera Array'],
             result_dict['Flame']['Heat Release Rate Cantera']] = heat_release_rate_extractor('Cantera',
                                                                                              plt_data=plt_data,
                                                                                              sort_arr=np.argsort(plt_data['boxlib', 'x']))

        # Heat Release Rate - PeleC
        if CHECK_FLAGS['Flame'].get('Heat Release Rate PeleC', False):
            try:
                [result_dict['Flame']['Heat Release Rate PeleC Array'],
                 result_dict['Flame']['Heat Release Rate PeleC']] = heat_release_rate_extractor('PeleC',
                                                                                                plt_data=plt_data,
                                                                                                sort_arr=np.argsort(plt_data['boxlib', 'x']))
            except:
                [result_dict['Flame']['Heat Release Rate PeleC Array'],
                 result_dict['Flame']['Heat Release Rate PeleC']] = heat_release_rate_extractor('Cantera',
                                                                                                plt_data=plt_data,
                                                                                                sort_arr=np.argsort(plt_data['boxlib', 'x']))

        # Flame Thickness/ Length
        if CHECK_FLAGS['Flame'].get('Flame Thickness', False) or CHECK_FLAGS['Flame'].get('Surface Length', False):
            [result_dict['Flame']['Surface Length'],
             result_dict['Flame']['Flame Thickness']] = flame_geometry_function(raw_data,
                                                                                domain_info,
                                                                                output_dir,
                                                                                CHECK_FLAGS)

    # Step 4: Maximum Pressure Processing
    if 'Maximum Pressure' in CHECK_FLAGS:
        # Step 4.1:
        result_dict['Maximum Pressure'] = {}

        # Position
        if CHECK_FLAGS['Maximum Pressure'].get('Position', False):
            [result_dict['Maximum Pressure']['Index'], result_dict['Maximum Pressure']['Position']] = wave_tracking('Maximum Pressure',
                                                                                                                    pre_loaded_data=pre_loaded_data)

        # Thermodynamic State
        if not CHECK_FLAGS['Maximum Pressure'].get('Position', False):
            result_dict['Maximum Pressure'] = {}
            [result_dict['Maximum Pressure']['Index'], result_dict['Maximum Pressure']['Position']] = wave_tracking("Maximum Pressure", pre_loaded_data=pre_loaded_data)

        # Determine the thermodynamic state ahead of the leading shock wave
        result_dict['Maximum Pressure']['Thermodynamic State'] = thermodynamic_state_extractor('Maximum Pressure',
                                                                                        plt_data=plt_data,
                                                                                        sort_arr=sort_arr,
                                                                                        pre_loaded_data=pre_loaded_data,
                                                                                        wave_loc=result_dict['Maximum Pressure']['Position'])

    # Step 5: Leading Shock Location Processing
    if 'Leading Shock' in CHECK_FLAGS:
        # Step 5.1
        result_dict['Leading Shock'] = {}

        # Position
        if CHECK_FLAGS['Leading Shock'].get('Position', False):
            [result_dict['Leading Shock']['Index'], result_dict['Leading Shock']['Position']] = wave_tracking('Leading Shock',
                                                                                                              pre_loaded_data=pre_loaded_data)

    # Step 6:
    if 'Pre-Shock' in CHECK_FLAGS:
        # Step 6.1:
        result_dict['Pre-Shock'] = {}

        # Step 6.2:
        if CHECK_FLAGS['Pre-Shock'].get('Thermodynamic State', False):
            # If the leading shock wave is not flagged, determine the location here
            if not CHECK_FLAGS['Leading Shock'].get('Position', False):
                result_dict['Leading Shock'] = {}
                [result_dict['Leading Shock']['Index'], _] = wave_tracking("Leading Shock", pre_loaded_data=pre_loaded_data)

            # Determine the thermodynamic state ahead of the leading shock wave
            result_dict['Pre-Shock']['Thermodynamic State'] = thermodynamic_state_extractor('Pre-Shock',
                                                                                            plt_data=plt_data,
                                                                                            sort_arr=sort_arr,
                                                                                            pre_loaded_data=pre_loaded_data,
                                                                                            wave_loc=result_dict['Leading Shock']['Position'])

    # Step 7:
    if 'Post-Shock' in CHECK_FLAGS:
        # Step 7.1:
        result_dict['Post-Shock'] = {}

        # Step 7.2:
        if CHECK_FLAGS['Post-Shock'].get('Thermodynamic State', False):
            # If the leading shock wave is not flagged, determine the location here
            if not CHECK_FLAGS['Leading Shock'].get('Position', False):
                result_dict['Leading Shock'] = {}
                [result_dict['Leading Shock']['Index'], _] = wave_tracking("Leading Shock", pre_loaded_data=pre_loaded_data)

            # Determine the thermodynamic state behind of the leading shock wave
            result_dict['Post-Shock']['Thermodynamic State'] = thermodynamic_state_extractor('Post-Shock',
                                                                                            plt_data=plt_data,
                                                                                            sort_arr=sort_arr,
                                                                                            pre_loaded_data=pre_loaded_data,
                                                                                            wave_loc=result_dict['Leading Shock']['Position'])

    # Step 8:
    if 'Domain State Animations' in CHECK_FLAGS:
        for key, value in CHECK_FLAGS.get('Domain State Animations', {}).items():
            if key != 'Surface Contour' and key != 'Flame Thickness':
                if key == 'Combined':
                    if isinstance(value, (np.ndarray, list, tuple)):
                        bool_check = True
                        var_name = value[0]
                        pele_name = value[1]
                    else:
                        bool_check = value
                        var_name = key

                    if bool_check:
                        x_data_arr = []
                        y_data_arr = []
                        y_lim = []
                        for i in range(len(value[0])):
                            # Collect the plot data
                            if pre_loaded_data is not None:
                                # x_data_arr.append(pre_loaded_data[np.argwhere(pre_loaded_data[0] == 'Position')[0][0]].compute())
                                x_data_arr.append(pre_loaded_data[np.argwhere(pre_loaded_data[0] == 'Position')[0][0]])
                                if var_name[i] not in pre_loaded_data[0]:
                                    # y_data_arr.append(pre_loaded_data[np.argwhere(pre_loaded_data[0] == var_name[i])[0][0]].compute())
                                    y_data_arr.append(pre_loaded_data[np.argwhere(pre_loaded_data[0] == var_name[i])[0][0]])
                                else:
                                    if key == 'Heat Release Rate Cantera':
                                        y_data_arr.append(result_dict['Flame']['Heat Release Rate Cantera Array'])
                                    else:
                                        y_data_arr.append(plt_data['boxlib', pele_name[i]][sort_arr].to_value())
                            else:
                                x_data_arr.append(plt_data['boxlib', 'x'][sort_arr].to_value())
                                y_data_arr.append(plt_data['boxlib', pele_name[i]][sort_arr].to_value())

                            # Collect the plot bounds based on the current desire
                            bnd_arr_index = [item[0] for item in animation_bnds].index(var_name[i])
                            y_lim.append([animation_bnds[bnd_arr_index][1], animation_bnds[bnd_arr_index][2]])

                        # Create directory for plt files
                        combined_key = "-".join(var_name)
                        temp_plt_dir = ensure_long_path_prefix(
                            os.path.join(output_dir, f"Animation-Frames", f"{combined_key}-Plt-Files"))

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
                        # Collect the plot data
                        if pre_loaded_data is not None:
                            # x_data_arr = pre_loaded_data[np.argwhere(pre_loaded_data[0] == 'Position')[0][0]].compute()
                            x_data_arr = pre_loaded_data[np.argwhere(pre_loaded_data[0] == 'Position')[0][0]]
                            if key not in pre_loaded_data[0]:
                                # y_data_arr = pre_loaded_data[np.argwhere(pre_loaded_data[0] == var_name)[0][0]].compute()
                                y_data_arr = pre_loaded_data[np.argwhere(pre_loaded_data[0] == var_name)[0][0]]
                            else:
                                if key == 'Heat Release Rate Cantera':
                                    y_data_arr = result_dict['Flame']['Heat Release Rate Cantera Array']
                                else:
                                    y_data_arr = plt_data['boxlib', pele_name][sort_arr].to_value()
                        else:
                            x_data_arr = plt_data['boxlib', 'x'][sort_arr].to_value()
                            y_data_arr = plt_data['boxlib', pele_name][sort_arr].to_value()

                        # Collect the plot bounds based on the current desire
                        bnd_arr_index = [item[0] for item in animation_bnds].index(var_name)
                        y_lim = [animation_bnds[bnd_arr_index][1], animation_bnds[bnd_arr_index][2]]

                        # Create directory for plt files
                        temp_plt_dir = ensure_long_path_prefix(
                            os.path.join(output_dir, f"Animation-Frames", f"{var_name}-Plt-Files"))

                if bool_check:
                    if os.path.exists(temp_plt_dir) is False:
                        os.makedirs(temp_plt_dir, exist_ok=True)

                    state_animation(method='Plot',
                                    time=time,
                                    x_data_arr=x_data_arr,
                                    y_data_arr=y_data_arr,
                                    y_bounds=y_lim,
                                    domain_size=domain_info,
                                    var_name=var_name,
                                    output_dir_path=temp_plt_dir)

    # Step 9:
    if 'Local State Animation' in CHECK_FLAGS:
        #
        local_physical_window = CHECK_FLAGS['Local State Animations']['Physical Window']
        # Step 1: Determine the location of the wave to be tracked
        if 'Index' in CHECK_FLAGS['Local State Animations']['Wave of Interest']:
            wave_idx = result_dict[f'{CHECK_FLAGS['Local State Animations']['Wave of Interest']}']['Index']
        else:
            wave_idx = wave_tracking(plt_data, sort_arr, pre_loaded_data=pre_loaded_data)

        for key, value in CHECK_FLAGS.get('Domain State Animations', {}).items():
            if key != 'Surface Contour' and key != 'Flame Thickness':
                if isinstance(value, (np.ndarray, list, tuple)):
                    bool_check = True
                    var_name = value[0]
                    pele_name = value[1]
                else:
                    bool_check = value
                    var_name = key
                    pele_name = key

                if bool_check:
                    # Collect the plot data
                    if pre_loaded_data is not None:
                        #x_data_arr = pre_loaded_data[np.argwhere(pre_loaded_data[0] == 'Position')[0][0]].compute()
                        x_data_arr = pre_loaded_data[np.argwhere(pre_loaded_data[0] == 'Position')[0][0]]
                        if key not in pre_loaded_data[0]:
                            #y_data_arr = pre_loaded_data[np.argwhere(pre_loaded_data[0] == var_name)[0][0]].compute()
                            y_data_arr = pre_loaded_data[np.argwhere(pre_loaded_data[0] == var_name)[0][0]]
                        else:
                            if key == 'Heat Release Rate Cantera':
                                y_data_arr = result_dict['Flame']['Heat Release Rate Cantera Array']
                            else:
                                y_data_arr = plt_data['boxlib', pele_name][sort_arr].to_value()
                    else:
                        x_data_arr = plt_data['boxlib', 'x'][sort_arr].to_value()
                        y_data_arr = plt_data['boxlib', pele_name][sort_arr].to_value()

                    # Collect the plot bounds based on the current desire
                    bnd_arr_index = [item[0] for item in animation_bnds].index(var_name)
                    x_lim = [np.searchsorted(x_data_arr / 100, wave_idx - local_physical_window, side='left'),
                             np.searchsorted(x_data_arr / 100, wave_idx + local_physical_window, side='right')]
                    y_lim = [animation_bnds[bnd_arr_index][1], animation_bnds[bnd_arr_index][2]]

                    # Create directory for plt files
                    temp_plt_dir = ensure_long_path_prefix(
                        os.path.join(output_dir,
                                     f"Animation-Frames",
                                     f"{'-'.join(['Local', CHECK_FLAGS['Local State Animations']['Wave of Interest'], var_name])}-Plt-Files"))

                    if os.path.exists(temp_plt_dir) is False:
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
                # Access the sub-dictionary
                sub_dict = CHECK_FLAGS[key]

                # Loop through the sub-dictionary to create the header array
                for sub_key, sub_value in sub_dict.items():
                    if sub_value is True:  # Check if the value is True
                        if sub_key == 'Thermodynamic State':
                            header_data.extend([f"{key} Temperature [K]", f"{key} Pressure [Pa]",
                                                f"{key} Density [kg/m^3]", f"{key} Soundspeed [m/s]"])
                        else:
                            if sub_key == 'Position' or sub_key == 'Flame Thickness' or sub_key == 'Surface Length':
                                unit_str = 'm'
                            elif sub_key == 'Velocity' or 'Relative Velocity':
                                unit_str = 'm/s'
                            elif sub_key == 'Heat Release Rate Cantera':
                                unit_str = 'W/m^3'
                            elif sub_key == 'Heat Release Rate PeleC':
                                unit_str = 'erg/cm3'
                            else:
                                unit_str = '[]'

                            header_data.extend([f"{key} {sub_key} [{unit_str}]"])

        # Step 3:
        with open(file_path, "w") as outfile:
            outfile.write("#")
            for i in range(len(header_data)):
                outfile.write("{0:<55.0f} ".format(int(i + 1)))
            outfile.write("\n#")
            for i in range(len(header_data)):
                outfile.write("{0:<55s} ".format(header_data[i]))
            outfile.write("\n")

            if not smoothing_check:
                for i in range(len(collective_results['Time']['Value'])):
                    # Write time value
                    outfile.write(" {0:<55e}".format(collective_results['Time']['Value'][i]))

                    #
                    for key in CHECK_FLAGS.keys():
                        if key in {'Flame', 'Leading Shock', 'Maximum Pressure', 'Pre-Shock', 'Post-Shock'}:
                            # Access the sub-dictionary
                            sub_dict = CHECK_FLAGS[key]

                            # Loop through the sub-dictionary to create the header array
                            for sub_key, sub_value in sub_dict.items():
                                if sub_value is True:  # Check if the value is True
                                    if sub_key == 'Thermodynamic State':
                                        for j in range(len(collective_results[key]['Thermodynamic State'][0])):
                                            outfile.write(" {0:<55e}".format(collective_results[key][sub_key][i][j]))
                                    else:
                                        outfile.write(" {0:<55e}".format(collective_results[key][sub_key][i]))

                    outfile.write("\n")

            else:
                for i in range(len(collective_results['Time']['Smooth']['Value'])):
                    # Write time value
                    outfile.write(" {0:<55e}".format(collective_results['Time']['Smooth']['Value'][i]))

                    #
                    for key in CHECK_FLAGS.keys():
                        if key in {'Flame', 'Leading Shock', 'Maximum Pressure', 'Pre-Shock', 'Post-Shock'}:
                            # Access the sub-dictionary
                            sub_dict = CHECK_FLAGS[key]

                            # Loop through the sub-dictionary to create the header array
                            for sub_key, sub_value in sub_dict.items():
                                if sub_value is True:  # Check if the value is True
                                    if sub_key == 'Thermodynamic State':
                                        for j in range(len(collective_results[key]['Smooth']['Thermodynamic State'][0])):
                                            outfile.write(" {0:<55e}".format(collective_results[key]['Smooth'][sub_key][i][j]))
                                    else:
                                        outfile.write(" {0:<55e}".format(collective_results[key]['Smooth'][sub_key][i]))

                    outfile.write("\n")

        outfile.close()
        return
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
    collective_results = {}
    for master_key in pelec_data[0]:
        print(master_key)
        print(collective_results)
        if master_key not in collective_results:
            collective_results[master_key] = {}
        for slave_key in pelec_data[0][master_key]:
            # Step 2.1
            temp_vec = []
            for i in range(len(pelec_data)):
                temp_vec.append(pelec_data[i][master_key][slave_key])
            # Step 2.2
            collective_results[master_key][slave_key] = temp_vec

    # Step 3:
    if 'Flame' in CHECK_FLAGS:
        print('Starting Flame Processing')
        # Step 3.1: Velocity
        if CHECK_FLAGS['Flame'].get('Velocity', False):
            collective_results['Flame']['Velocity'] = np.gradient(
                collective_results['Flame']['Position']) / np.gradient(collective_results['Time']['Value'])
        # Step 3.2: Relative Velocity
        if CHECK_FLAGS['Flame'].get('Relative Velocity', False):
            if CHECK_FLAGS['Flame'].get('Velocity', False):
                collective_results['Flame']['Relative Velocity'] = collective_results['Flame']['Velocity'] - \
                                                                   collective_results['Flame']['Gas Velocity']
            else:
                print('ERROR: Must Enable Velocity Flag to compute the relative velocity')

        # Step 3.3: Data Smoothing
        if CHECK_FLAGS['Flame'].get('Smoothing', False) and CHECK_FLAGS['Flame'].get('Position', False):
            data_smoothing_function(collective_results, 'Flame', 'Position')

        if CHECK_FLAGS['Flame'].get('Smoothing', False) and CHECK_FLAGS['Flame'].get('Velocity', False):
            data_smoothing_function(collective_results, 'Flame', 'Velocity')

        if CHECK_FLAGS['Flame'].get('Smoothing', False) and CHECK_FLAGS['Flame'].get('Relative Velocity', False):
            data_smoothing_function(collective_results, 'Flame', 'Relative Velocity')

        if CHECK_FLAGS['Flame'].get('Smoothing', False) and CHECK_FLAGS['Flame'].get('Thermodynamic State', False):
            data_smoothing_function(collective_results, 'Flame', 'Thermodynamic State')

        if CHECK_FLAGS['Flame'].get('Smoothing', False) and (
        CHECK_FLAGS['Flame'].get('Heat Release Rate Cantera', False)):
            data_smoothing_function(collective_results, 'Flame', 'Heat Release Rate Cantera')

        if CHECK_FLAGS['Flame'].get('Smoothing', False) and (
        CHECK_FLAGS['Flame'].get('Heat Release Rate PeleC', False)):
            data_smoothing_function(collective_results, 'Flame', 'Heat Release Rate PeleC')

        if CHECK_FLAGS['Flame'].get('Smoothing', False) and CHECK_FLAGS['Flame'].get('Flame Thickness', False):
            data_smoothing_function(collective_results, 'Flame', 'Flame Thickness')

        if CHECK_FLAGS['Flame'].get('Smoothing', False) and CHECK_FLAGS['Flame'].get('Surface Length', False):
            data_smoothing_function(collective_results, 'Flame', 'Surface Length')

        print('Completed Flame Processing')
    # Step 4: Leading Shock Processing
    if 'Leading Shock' in CHECK_FLAGS:
        print('Starting Leading Shock Processing')
        # Step 4.1: Velocity
        if CHECK_FLAGS['Leading Shock'].get('Velocity', False):
            collective_results['Leading Shock']['Velocity'] = np.gradient(
                collective_results['Leading Shock']['Position']) / np.gradient(collective_results['Time']['Value'])
        # Step 4.1: Data Smoothing
        if CHECK_FLAGS['Leading Shock'].get('Smoothing', False) and CHECK_FLAGS['Leading Shock'].get('Position',
                                                                                                     False):
            data_smoothing_function(collective_results, 'Leading Shock', 'Position')

        if CHECK_FLAGS['Leading Shock'].get('Smoothing', False) and CHECK_FLAGS['Leading Shock'].get('Velocity',
                                                                                                     False):
            data_smoothing_function(collective_results, 'Leading Shock', 'Velocity')

        print('Completed Leading Shock Processing')
    # Step 5: Maximum Pressure Processing
    if 'Maximum Pressure' in CHECK_FLAGS:
        print('Starting Max Pressure Processing')
        # Step 5.1: Data Smoothing
        if CHECK_FLAGS['Maximum Pressure'].get('Smoothing', False) and CHECK_FLAGS['Maximum Pressure'].get(
                'Position', False):
            data_smoothing_function(collective_results, 'Maximum Pressure', 'Position')

        if CHECK_FLAGS['Maximum Pressure'].get('Smoothing', False) and CHECK_FLAGS['Maximum Pressure'].get(
                'Thermodynamic State', False):
            data_smoothing_function(collective_results, 'Maximum Pressure', 'Thermodynamic State')

        print('Completed Max Pressure Processing')
    # Step 6: Pre-Shock Processing
    if 'Pre-Shock' in CHECK_FLAGS:
        print('Starting Pre-Shock Processing')
        # Step 6.1: Data Smoothing
        if CHECK_FLAGS['Pre-Shock'].get('Smoothing', False) and CHECK_FLAGS['Pre-Shock'].get('Thermodynamic State',
                                                                                             False):
            data_smoothing_function(collective_results, 'Pre-Shock', 'Thermodynamic State')

        print('Completed Pre-Shock Processing')
    # Step 7: Post-Shock Processing
    if 'Post-Shock' in CHECK_FLAGS:
        print('Starting Post-Shock Processing')
        # Step 7.1: Data Smoothing
        if CHECK_FLAGS['Post-Shock'].get('Smoothing', False) and CHECK_FLAGS['Post-Shock'].get(
                'Thermodynamic State', False):
            data_smoothing_function(collective_results, 'Post-Shock', 'Thermodynamic State')

        print('Completed Post-Shock Processing')

    # Step 8: Write to file, if any of the sub-dictionary values except 'Domain State Animations' are true
    print('Start Output File Writing')
    write_to_file = False
    smoothing_flag = False
    for key, sub_dict in CHECK_FLAGS.items():
        if 'Smoothing' in sub_dict and sub_dict['Smoothing'] == True:
            smoothing_flag = True
        if key != 'Domain State Animations':
            if any(sub_dict.values()):  # If any value is True
                write_to_file = True
                break  # Break as soon as we find a True value

    if write_to_file:
        file_output(ensure_long_path_prefix(os.path.join(output_dir, 'Wave-Tracking-Results.txt')), False)
        if smoothing_flag:
            file_output(ensure_long_path_prefix(os.path.join(output_dir, 'Wave-Tracking-Smooth-Results.txt')), True)

        for master_key, sub_dict in CHECK_FLAGS.items():
            if 'Smoothing' in sub_dict and sub_dict['Smoothing'] == True:
                file_output(ensure_long_path_prefix(os.path.join(output_dir, 'Wave-Tracking-Smooth-Results.txt')),
                            True)
    print('Completed Output File Writing')

    # Step 7: Create Variable Evolution
    print('Starting Animation Processing')
    if 'Domain State Animations' in CHECK_FLAGS:
        for key, value in CHECK_FLAGS.get('Domain State Animations', {}).items():
            if isinstance(value, (np.ndarray, list, tuple)):
                bool_check = value[0]
                # Create directory for plt files
                if key == 'Combined':
                    plot_key = "-".join(value[0])
                else:
                    plot_key = key
            else:
                plot_key = key
                bool_check = value

            if bool_check:
                # Create directory for plt files
                temp_plt_dir = ensure_long_path_prefix(
                    os.path.join(output_dir, f"Animation-Frames", f"{plot_key}-Plt-Files"))

                state_animation(
                    method='Animate',
                    folder_path=temp_plt_dir,
                    animation_filename=ensure_long_path_prefix(
                        os.path.join(output_dir, f"{plot_key}-Evolution-Animation.mp4")),
                )

    if 'Local State Animations' in CHECK_FLAGS:
        for key, value in CHECK_FLAGS.get('Domain State Animations', {}).items():
            if key != 'Combined':
                if isinstance(value, (np.ndarray, list, tuple)):
                    bool_check = value[0]
                else:
                    bool_check = value

                if bool_check:
                    temp_plt_dir = ensure_long_path_prefix(os.path.join(output_dir, f"Animation-Frames",
                                                                        f"Local-{CHECK_FLAGS['Local State Animations']['Wave of Interest']}-{key}-Plt-Files"))

                    state_animation(
                        method='Animate',
                        folder_path=temp_plt_dir,
                        animation_filename=ensure_long_path_prefix(
                            os.path.join(output_dir,
                                         f"Local-{CHECK_FLAGS['Local State Animations']['Wave of Interest']}-{key}-Evolution-Animation.mp4")),
                    )

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
    # Step 1: Set all the desired tasks to be performed bny the python script
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
            'Velocity': (False, 'x_velocity'),
            'Species': False,
            'Heat Release Rate Cantera': False,
            'Heat Release Rate PeleC': (False, 'heatRelease'),
            'Surface Contour': False,
            'Flame Thickness': False,
            'Combined': (('Temperature', 'Pressure'), ('Temp', 'pressure'))
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

    time_data_dir = []
    for raw_data_folder in os.listdir(dir_path):
        if os.path.isdir(os.path.join(dir_path, raw_data_folder)) and raw_data_folder.startswith('Raw-PeleC'):
            raw_data_dir = os.path.join(dir_path, raw_data_folder)
            for time_step in os.listdir(raw_data_dir):
                if os.path.isdir(os.path.join(raw_data_dir, time_step)) and time_step.startswith('plt'):
                    time_data_dir.append(os.path.join(raw_data_dir, time_step))

    # Step 4: Chronologically order the pltFiles and truncate the raw data list if skip loading is enabled
    updated_data_list = sort_files(time_data_dir)

    # Step 5: Determine the domain sizing parameters (size, # of cells)
    if row_idx == 'DDT':
        domain_info = domain_size_parameters(
            os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Raw-PeleC-Data', ddt_plt_file), row_idx)
    else:
        domain_info = domain_size_parameters(updated_data_list[0], row_idx)

    # Step 6: Create the result directories
    if os.path.exists(os.path.join(dir_path, f"Processed-Global-Results")) is False:
        os.mkdir(os.path.join(dir_path, f"Processed-Global-Results"))
    output_dir_path = os.path.join(dir_path, f"Processed-Global-Results", f"y-{domain_info[1][0][1]:.3g}cm")

    # Step 7:

    animation_axis_bnds = animation_axis(updated_data_list, ddt_plt_file, domain_info, CHECK_FLAGS)

    # Step 8:
    print('Begining PeleC Processing')
    pelec_processing(updated_data_list, domain_info, animation_axis_bnds, output_dir_path, CHECK_FLAGS)
    print('Completed PeleC Processing')

    end_time = time.time()
    print(f"Execution time: {end_time - start_time} seconds")
    return

if __name__ == '__main__':
    cProfile.run('main()')