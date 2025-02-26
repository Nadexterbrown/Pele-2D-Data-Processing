import os, multiprocessing, re, itertools, traceback, sys, difflib
import pandas as pd
import numpy as np
import numpy.polynomial.polynomial as poly
import matplotlib.pyplot as plt

########################################################################################################################
# Global Program Setting Variables
########################################################################################################################

version = 1
n_procs = 1

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
        print('Performing Processing in Parallel')
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
        print('Performing Processing in Serial')
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

########################################################################################################################
# Function Scripts - Data Read/Write
########################################################################################################################

def data_import(file_path):
    # Read the file, extracting the second row as column names
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Extract the header row (second line) and clean column names
    raw_headers = lines[1].strip()

    # Use regex to split headers by multiple spaces while keeping words together
    headers = re.split(r'\s{2,}', raw_headers)  # Split on 2+ spaces
    headers[0] = headers[0].lstrip('#')  # Remove the '#' from the first column name

    # Read the data, skipping the first two rows to exclude the initial comment and headers
    return pd.read_csv(file_path, delim_whitespace=True, skiprows=2, names=headers)

def data_export(file_name, data):
    """
            Save the smoothed data to a file while preserving column orientation,
            ensuring columns are 55 characters wide, and using the same header as the input file.

            Parameters:
                data (pd.DataFrame): Smoothed data.
                output_file (str): Path to save the output file.
            """
    # Ensure that the data is written with exactly 55-character wide columns
    with open(file_name, 'w') as f:
        # Write the header
        f.write(' '.join([f"{col:<55}" for col in data.columns]) + '\n')

        # Write the data
        for _, row in data.iterrows():
            f.write(' '.join([f"{val:<55}" for val in row]) + '\n')

    return

def find_closest_columns(df, search_string, num_matches=1):
    """
    Finds the closest matching column names in a DataFrame to a given search string.

    Parameters:
    - df: pandas DataFrame.
    - search_string: string to match against column headers.
    - num_matches: number of closest matches to return.

    Returns:
    - List of column names that best match the search string.
    """
    columns = df.columns
    matches = difflib.get_close_matches(search_string, columns, n=num_matches, cutoff=0.2)
    return matches

########################################################################################################################
# Function Scripts - Down Sampling
########################################################################################################################

def down_sampling(data, dt):
    ###########################################
    # Main Function
    ###########################################

    # Step 1: Extract the time vector and determine the max time
    time_vec = data[find_closest_columns(data, 'Time')].values.ravel()
    t_f = np.max(time_vec)

    # Step 2: Create an empty DataFrame to store the copied rows
    rows_to_copy = []

    # Step 3: Loop over time and store the times nearest the current time step
    t = np.min(time_vec)
    while t < t_f:
        # Search the time vector for the nearest value (greater than) to t
        data_idx = np.searchsorted(time_vec, t, side='right')  # Find insertion index for larger value
        if data_idx > len(time_vec):
            raise ValueError(f"Time error, t >= {t} does not exist")

        # Copy the row at data_idx and add to the list
        rows_to_copy.append(data.iloc[[data_idx]].copy())

        # Increase the current time
        t += dt

    # Step 4: Concatenate all collected rows into a new DataFrame
    current_data = pd.concat(rows_to_copy, ignore_index=True)

    return current_data

########################################################################################################################
# Function Scripts -
########################################################################################################################

def data_smoothing(data, bin):
    ###########################################
    # Main Function
    ###########################################

    if bin % 2 == 0:
        raise ValueError("Bin size must be odd to center around a cell.")

    half_bin = bin // 2

    time_name = find_closest_columns(data, 'Time')[0]
    exclude_columns = {time_name,
                       find_closest_columns(data, 'Flame Heat Release Rate Cantera')[0],
                       find_closest_columns(data, 'Flame Heat Release Rate PeleC')[0]}
    name_columns = [col for col in data.columns if col not in exclude_columns]

    time_flag = True
    time_array = []
    smoothed_data = pd.DataFrame(columns=data.columns)
    for name in name_columns:
        smoothed_values = []
        for i in range(half_bin, len(data) - half_bin):  # Start from half_bin and end at len(data) - half_bin
            # Define the bin range
            start_idx = i - half_bin
            end_idx = i + half_bin + 1

            # Select data within the bin
            bin_data = data.iloc[start_idx:end_idx]

            tmp_time = bin_data[time_name].values.flatten()
            tmp_data = bin_data[name].values

            # Remove NaN values from temp_data (value_column) only

            valid_idx = ~np.isnan(tmp_data).flatten()
            tmp_time = tmp_time[valid_idx]
            tmp_data = tmp_data[valid_idx]

            # Perform the first-order polynomial fit if there are enough valid points
            if len(tmp_data) >= 2:
                x = tmp_time
                y = tmp_data
                try:
                    coeffs = np.polyfit(x, y, 1)
                except:
                    print('Fail')
                poly = np.poly1d(coeffs)

                # Get the smoothed value at the current time value
                smoothed_values.append(poly(data[time_name].iloc[i]))
            else:
                # If not enough valid points, append the original value
                smoothed_values.append(data[name].iloc[i])

            if time_flag:
                time_array.append(data[time_name].iloc[i])

        time_flag = False

        # Replace the original column with smoothed values
        smoothed_data[name] = smoothed_values

    # Add the time column to the smoothed data
    smoothed_data[time_name] = time_array

    return smoothed_data

def velocity_calculation(data, bin):
    ###########################################
    # Internal Function
    ###########################################
    def model_polynomial_fit(x, y, polyOrder):
        position_fit_coef = poly.polyfit(x, y, polyOrder)
        position_polyval = poly.polyval(x, position_fit_coef)
        position = poly.Polynomial(position_fit_coef)

        velocity_fit_coef = poly.polyder(position_fit_coef, 1)
        velocity_polyval = poly.polyval(x, velocity_fit_coef)
        velocity = poly.Polynomial(velocity_fit_coef)

        return position(x), velocity(x)

    ###########################################
    # Main Function
    ###########################################
    if bin % 2 == 0:
        raise ValueError("Bin size must be odd to center around a cell.")

    half_bin = bin // 2
    time, position, velocity = [], [], []
    for i in range(half_bin, len(data) - half_bin):  # Start from half_bin and end at len(data) - half_bin
        # Define the bin range
        start_idx = i - half_bin
        end_idx = i + half_bin + 1
        # Define temporary binned arrays
        tmp_time = data[find_closest_columns(data, 'Time')].values.flatten()[start_idx:end_idx]
        tmp_position = data[find_closest_columns(data, 'Flame Position')].values.flatten()[start_idx:end_idx]
        position_fit, velocity_fit = model_polynomial_fit(tmp_time, tmp_position, 2)

        time.append(data[find_closest_columns(data, 'Time')].values.flatten()[i])
        position.append(position_fit[half_bin])
        velocity.append(velocity_fit[half_bin])

    return np.array([time, position, velocity])

########################################################################################################################
# Function Scripts -
########################################################################################################################

def main():

    # Step 0: Script inputs
    down_sampling_dt = 1/160000 # Delta_t window used to down sample the numerical simulation results
    ddt_time = 8.228925e-04     # Relative time of DDT to that simulation run
    bin_window = 11

    # Step 1: Paths to directories where data files are stored
    #dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = r'Z:\Research\DDT-Simulations\PeleC-Simulations\2D-Channel-FA-DDT\H2-O2-Phi-1.0-P-10.0-bar-T-503-K\Level-6-External-Processing'
    file_name = 'Wave-Tracking-Results-V34.txt'
    files = [os.path.join(dir_path, 'Part-1', 'y-0.0462cm', file_name),
             os.path.join(dir_path, 'Part-2', 'y-0.0462cm', file_name),
             os.path.join(dir_path, 'Part-3', 'y-0.0462cm', file_name)]

    # Step 2: Load the data to memory
    data_arrs = np.empty(len(files), dtype=object)
    for i, data_file in enumerate(files):
        data_arrs[i] = data_import(data_file)
        if i > 0:
            tmp_val = data_arrs[i - 1][find_closest_columns(data_arrs[i - 1], 'Time')].values.ravel()[-1]
            data_arrs[i][find_closest_columns(data_arrs[i], 'Time')] += tmp_val
            if i > 1:
                ddt_time += tmp_val

    data = pd.concat(data_arrs, ignore_index=True)

    # Step 3: Apply down sampling
    data_ds = down_sampling(data, down_sampling_dt)

    # Step 4: Write the down sampled data to file
    output_dir = os.path.join(dir_path, 'Down-Sampled-Data')
    output_file_name = 'PeleC-Collective-Down-Sampled-Results.txt'
    if os.path.exists(output_dir) is False:
        os.mkdir(output_dir)

    data_export(os.path.join(output_dir, output_file_name), data_ds)

    # Step 4: Smooth the data similar to the approach used in experimental processing techniques
    data_smoothed = data_smoothing(data_ds, bin_window)

    output_dir = os.path.join(dir_path, 'Down-Sampled-Data')
    output_file_name = 'PeleC-Collective-Down-Sampled-Smoothed-Results.txt'
    if os.path.exists(output_dir) is False:
        os.mkdir(output_dir)

    data_export(os.path.join(output_dir, output_file_name), data_smoothed)

    # Step 5: Smoothed Velocity Calculation
    time, position, velocity = velocity_calculation(data_ds, bin_window)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(position * 100, velocity, 'k-')
    plt.title("Position vs Velocity")
    plt.legend()
    plt.show()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(time * 1000, velocity, 'k-')
    plt.title("Time vs Velocity")
    plt.legend()
    plt.show()

    return

if __name__ == "__main__":
    main()