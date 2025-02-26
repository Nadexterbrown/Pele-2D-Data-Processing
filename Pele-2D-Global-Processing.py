import os, yt, itertools, multiprocessing, textwrap, re

from numpy.fft.helper import fftfreq
from sklearn.neighbors import NearestNeighbors
from sdtoolbox.thermo import soundspeed_fr
import matplotlib.animation as animation
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize
import cantera as ct
import numpy as np
import pywt

yt.set_log_level(0)

n_proc = 24
bin_size = 51


class MyClass():
    pass


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


def smoothing_function(collective_results, master_key, slave_key):
    def polynomial_fit_over_array(x, y, bin_size=51, degree=1):
        """
        Fits a polynomial of the specified degree to the data within bins
        centered around each point in the input array, and evaluates the
        polynomial at each point.

        Parameters:
            x (list or numpy array): The x-coordinates of the data points.
            y (list or numpy array): The y-coordinates of the data points.
            bin_size (int): The size of the bin around each point.
            degree (int): The degree of the polynomial to fit.

        Returns:
            x_fit (numpy array): The x-coordinates of the data points matched to the output.
            values (numpy array): The evaluated values of the polynomial at each point.
        """
        values = []
        derivative = []
        x_fit = []

        half_bin = bin_size // 2

        for i, point in enumerate(x):
            if i >= half_bin and i <= (len(x) - half_bin):
                # Find the indices within the bin
                center_index = np.argmin(np.abs(x - point))
                start_index = int(i - half_bin)
                end_index = int(i + half_bin)

                # Fit a polynomial to the data within the bin
                x_bin = np.array(x[start_index:end_index], dtype=float)
                y_bin = np.array(y[start_index:end_index], dtype=float)

                coefficients = np.polyfit(x_bin, y_bin, degree)

                # Evaluate the polynomial at the given point
                value = np.polyval(coefficients, point)
                values.append(value)
                x_fit.append(point)

                # Calculate the derivative of the polynomial at the given point
                poly_derivative = np.polyder(coefficients)
                derivative_value = np.polyval(poly_derivative, point)
                derivative.append(derivative_value)

        return np.array(x_fit), np.array(values), np.array(derivative)

    """

    """
    # Step 1: Create the smooth sub-sub-dictonary within the results directory if one does not exist
    if not collective_results[master_key].get('Smooth', False):
        collective_results[master_key]["Smooth"] = {}

    # Step 2: Create a new sub-sub-sub dict pertaining to the desired variable
    collective_results[master_key]["Smooth"][slave_key] = {}

    # Step 3:
    if slave_key == 'Thermodynamic State':
        temp_var_arr = np.empty((len(collective_results[master_key][slave_key]), 4), dtype=object)
        for i in range(len(collective_results[master_key][slave_key])):
            temp_var_arr[i, 0] = collective_results[master_key][slave_key][i][0]
            temp_var_arr[i, 1] = collective_results[master_key][slave_key][i][1]
            temp_var_arr[i, 2] = collective_results[master_key][slave_key][i][2]
            temp_var_arr[i, 3] = collective_results[master_key][slave_key][i][3]

        collective_results[master_key]["Smooth"][slave_key] = np.empty(4, dtype=object)
        for i in range(len(collective_results[master_key]["Smooth"][slave_key])):
            [temp_time, temp_vec, _] = polynomial_fit_over_array(np.array(collective_results['Time']['Value']),
                                                                 np.array(temp_var_arr[:, i]), bin_size=bin_size)
            collective_results[master_key]["Smooth"][slave_key][i] = temp_vec
    else:
        [temp_time, temp_vec, _] = polynomial_fit_over_array(np.array(collective_results['Time']['Value']),
                                                             np.array(collective_results[master_key][slave_key]),
                                                             bin_size=bin_size)

        collective_results[master_key]["Smooth"][slave_key] = temp_vec

    if not collective_results['Time'].get('Smooth', False):
        collective_results['Time']["Smooth"] = {}
        collective_results['Time']['Smooth']['Value'] = temp_time

    return collective_results


def mechanism_species(mechanism_file):
    """

    Args:
        mechanism_file:

    Returns:

    """
    # Step 1: Create cantera gas object from the provided mechanism file
    gas = ct.Solution(mechanism_file)
    speces_names = gas.species_names
    del gas
    return speces_names


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

    # Step 3:
    if isinstance(desired_y_location, str) is True:
        if desired_y_location == "Bottom":
            y_slice_index = 0
            y_slice_loc = data.LeftEdge[1].to_value()
        elif desired_y_location == "Top":
            y_slice_index = data.ActiveDimensions[1] - 1
            y_slice_loc = data.RightEdge[1].to_value()
        else:
            y_slice_index = int((data.ActiveDimensions[1] / 2) - 1)
            y_slice_loc = data["boxlib", str('y')][0][y_slice_index].to_value()[0]
    else:
        y_slice_index = np.argwhere(data["boxlib", str('y')][0][:].to_value() <= desired_y_location)[-1][0]
        y_slice_loc = data["boxlib", str('y')][0][y_slice_index].to_value()[0]

    return (
    np.array([[int(data.ActiveDimensions[0]), int(y_slice_index)], [data.RightEdge[0].to_value(), y_slice_loc]]),
    np.array([[int(data.ActiveDimensions[0]), int(data.ActiveDimensions[1])],
              [data.RightEdge[0].to_value(), data.RightEdge[1].to_value()]]))


def wave_tracking_function(raw_data, sort_arr, tracking_str, wave_type):
    """

    Args:
        raw_data:
        tracking_str:
        wave_type:

    Returns:

    """
    # Step 1: Load the desired marker and x positions
    temp_data = raw_data["boxlib", str(tracking_str)][sort_arr].to_value()
    temp_x_pos = raw_data["boxlib", str('x')][sort_arr].to_value()
    # Step 2:
    if wave_type == "Flame":
        wave_index = np.argwhere(temp_data >= 2000)[-1][0]
    elif wave_type == "Maximum Pressure":
        temp_index = temp_data
        wave_index = np.argwhere(temp_index == np.max(temp_index))[-1][0]
    elif wave_type == "Leading Shock":
        # From the farthest right point in the domain, determine the leading pressure wave location by 1% increase from this value
        pressure_baseline = temp_data[-1]
        try:
            wave_index = np.argwhere(temp_data >= 1.01 * pressure_baseline)[-1][0]
        except:
            wave_index = 0
    # Step 3:
    plt_check = False
    if plt_check:
        plt.figure(figsize=(8, 6))
        plt.plot(temp_x_pos, temp_data, linestyle='-', color='k')
        plt.plot(temp_x_pos[wave_index], temp_data[wave_index], marker='.', linestyle='-', color='r')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f"{tracking_str}")
        plt.show()

    return wave_index, temp_x_pos[wave_index] / 100


def thermodynamic_state_function(raw_data, sort_arr, wave_idx, wave_type, input_params):
    # Step 1: Collect the position of the wave
    wave_pos = raw_data["boxlib", str('x')][sort_arr].to_value()[wave_idx]

    # Step 2:
    if wave_type == "Flame":
        probe_index = np.argwhere(raw_data["boxlib", str('x')][sort_arr].to_value() >= wave_pos + (-0.001))[0][0]
    elif wave_type == "Maximum Pressure":
        try:
            probe_index = np.argwhere(raw_data["boxlib", str('x')][sort_arr].to_value() >= wave_pos)[0][0]
        except:
            probe_index = len(raw_data["boxlib", str('x')][sort_arr]) - 1
    elif wave_type == "Pre-Shock" or wave_type == "Post-Shock":
        if wave_type == "Pre-Shock":
            try:
                probe_index = np.argwhere(raw_data["boxlib", str('x')][sort_arr].to_value() >= wave_pos + (1 / 10))[0][
                    0]
            except:
                probe_index = len(raw_data["boxlib", str('x')][sort_arr]) - 1

        elif wave_type == "Post-Shock":
            try:
                probe_index = np.argwhere(raw_data["boxlib", str('x')][sort_arr].to_value() >= wave_pos - (1 / 10))[0][
                    0]
            except:
                probe_index = len(raw_data["boxlib", str('x')][sort_arr]) - 1

    # Step 3: Extract the state variables from the raw data
    temperature = raw_data["boxlib", str('Temp')][sort_arr][probe_index].to_value()
    pressure = raw_data["boxlib", str('pressure')][sort_arr][probe_index].to_value()

    species_comp = {}
    for i in range(len(input_params.species)):
        species_comp.update({str(input_params.species[i]):
                                 raw_data["boxlib", str("Y(" + input_params.species[i] + ")")][sort_arr][
                                     probe_index].to_value()})

    # Step 3: Create a Cantera object to extract soundspeed
    gas_obj = ct.Solution(input_params.mech)
    gas_obj.TPY = (temperature,
                   pressure / 10,
                   species_comp)

    # Step 4:
    result_array = np.zeros(4, dtype=float)
    result_array[0] = gas_obj.T
    result_array[1] = gas_obj.P
    result_array[2] = gas_obj.density_mass
    result_array[3] = soundspeed_fr(gas_obj)
    del gas_obj

    return result_array.tolist()


def heat_release_rate_function(raw_data, sort_arr, input_params):
    # Step 1: Collect the partial molar enthalpies and net production rates
    net_production_rates = np.empty(len(raw_data["boxlib", str('x')][sort_arr].to_value()), dtype=object)
    partial_molar_enthalpies = np.empty(len(raw_data["boxlib", str('x')][sort_arr].to_value()), dtype=object)
    for i in range(len(raw_data["boxlib", str('x')][sort_arr].to_value())):
        species_comp = {}
        temp_obj = ct.Solution(input_params.mech)
        for j in range(len(temp_obj.species_names)):
            species_comp.update({str(temp_obj.species_names[j]):
                                     raw_data["boxlib", str("Y(" + temp_obj.species_names[j] + ")")][sort_arr][
                                         i].to_value()})

        temp_obj.TPY = (raw_data["boxlib", str("Temp")][sort_arr][i].to_value(),
                        raw_data["boxlib", str("pressure")][sort_arr][i].to_value() * 10,
                        species_comp)

        net_production_rates[i] = temp_obj.net_production_rates
        partial_molar_enthalpies[i] = temp_obj.partial_molar_enthalpies

        del temp_obj

    # Step 3:
    heat_release_rate = np.zeros(len(raw_data["boxlib", str('x')][sort_arr].to_value()))
    for i in range(len(raw_data["boxlib", str('x')][sort_arr].to_value())):
        integral = 0
        for j in range(len(input_params.species)):
            integral = integral + net_production_rates[i][j] * partial_molar_enthalpies[i][j]

        heat_release_rate[i] = -1.0 * integral

    return heat_release_rate, np.max(heat_release_rate)


def flame_thickness_function(raw_data, sort_arr, wave_idx):
    # Step 1: Load the desired marker and x positions
    if wave_idx < 100:
        temp_data = raw_data["boxlib", str('Temp')][sort_arr].to_value()[0:2 * wave_idx]
        temp_x_pos = raw_data["boxlib", str('x')][sort_arr].to_value()[0:2 * wave_idx] / 100
    else:
        temp_data = raw_data["boxlib", str('Temp')][sort_arr].to_value()[wave_idx - 100:wave_idx + 100]
        temp_x_pos = raw_data["boxlib", str('x')][sort_arr].to_value()[wave_idx - 100:wave_idx + 100] / 100

    # Step 2: Compute the gradient of temperature with respect to position
    temperature_gradient = abs(np.gradient(temp_data) / np.gradient(temp_x_pos))
    temperature_gradient_idx = np.argwhere(temperature_gradient == np.max(temperature_gradient))[0][0]

    # Step 3:
    flame_thickness = (np.max(temp_data) - np.min(temp_data)) / np.max(temperature_gradient)

    # Step 4:
    flame_thickness_arr = [temp_x_pos[temperature_gradient_idx] - flame_thickness / 2,
                           temp_x_pos[temperature_gradient_idx] + flame_thickness / 2]

    plt_check = False
    if plt_check:
        plt.figure(figsize=(8, 6))
        plt.plot(temp_x_pos, temp_data, linestyle='-', color='k')
        plt.plot(flame_thickness_arr, [2000, 2000], linestyle='-', color='r')
        plt.plot(temp_x_pos[temperature_gradient_idx], 2000, marker='.', markersize=10, color='b')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xlim([temp_x_pos[temperature_gradient_idx] - flame_thickness,
                  temp_x_pos[temperature_gradient_idx] + flame_thickness])
        plt.show()

    return flame_thickness


def flame_length_contour_function(raw_data, plt_data, sort_arr, plt_check=False):
    def sort_by_nearest_neighbors(points, plt_check=False):
        # Convert points to numpy array if it's a list of lists
        points = np.array(points)

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
        distance_arr = []

        for i in range(1, len(points)):
            neighbors = []
            temp_idx = np.argwhere(indices[:, 0] == order[i - 1])[0][0]
            for neighbor_idx in indices[temp_idx, 1::]:
                if neighbor_idx not in order and neighbor_idx != indices[temp_idx, 0]:
                    neighbors.append(neighbor_idx)
                    # Mark the neighbor as visited
                    order.append(neighbor_idx)
                    distance_arr.append(distances[temp_idx, np.argwhere(indices[temp_idx, :] == neighbor_idx)])
                    # Stop if we've found enough neighbors for the given point
                    break
        # Check surface length from distances for comparison
        surface_approx = np.sum(distance_arr)
        # print('Approximate Surface Length =', surface_approx, 'cm')
        return points[order]

    def flame_front_wavelength_spectrum(raw_data, flame_contours):
        def segment_flame_contour(x, y, plt_check=False):
            # Step 1: Compute the difference in x-values
            dx = np.diff(x)
            # Step 2: Identify points where x-values turn back (i.e., sign change in dx)
            turning_points = np.where(np.diff(np.sign(dx)))[0] + 1  # +1 to correct index after np.diff
            # Step 3: Split the data into segments at turning points
            segments = []
            start = 0
            for tp in turning_points:
                segments.append((x[start:tp], y[start:tp]))
                start = tp
            # Append the last segment
            segments.append((x[start:], y[start:]))

            if plt_check:
                plt.figure(figsize=(8, 6))
                for i in range(len(segments)):
                    plt.plot(segments[i][0], segments[i][1])
                plt.xlabel('Wavelength (m)')
                plt.ylabel('Amplitude')
                plt.title('FFT Amplitude vs Wavelength')
                plt.grid(True)
                plt.show()

            return segments

        def contour_poly_fit_and_distance(x_data, y_data, degree=2, plt_check=False):
            # Step 1: Create a polynomial fit of the provided data
            coefficients = np.polyfit(x_data, y_data, degree)
            poly_func = np.poly1d(coefficients)

            def distance_to_curve(x0, y0):
                # Step 1: Minimize the distance function to find the closest point on the polynomial
                def distance(x):
                    y = poly_func(x)
                    return np.sqrt((x - x0) ** 2 + (y - y0) ** 2)

                # Step 2: Optimize to find the x that minimizes the distance
                result = minimize(distance, x0)  # Initial guess is x0

                # Step 3: Closest point on the curve
                x_closest = result.x[0]
                y_closest = poly_func(x_closest)

                # Step 4: Calculate the distance
                distance_val = np.sqrt((x_closest - x0) ** 2 + (y_closest - y0) ** 2)
                return distance_val, (x_closest, y_closest)

            # Step 2: Calculate distances for each data point
            distances = []
            closest_points = []

            for x0, y0 in zip(x_data, y_data):
                dist, point = distance_to_curve(x0, y0)
                distances.append(dist)
                closest_points.append(point)

            if plt_check:
                # Step : Create the resulting figure
                plt.figure(figsize=(12, 10))
                # Plot the original data and wavelet reconstruction
                plt.subplot(2, 1, 1)
                plt.plot(x_data, y_data, label='Original')
                plt.plot(np.linspace(0, data.RightEdge.to_value()[1], len(x_data)),
                         poly_func(np.linspace(0, data.RightEdge.to_value()[1], len(x_data))),
                         label='Polynomial Fit', linestyle='dashed')
                plt.legend()
                plt.grid()
                # Plot the distance
                plt.subplot(2, 1, 2)
                plt.plot(np.arange(len(x_data)), distances, label='Original')
                plt.legend()
                plt.grid()
                plt.show()

            return distances, closest_points, poly_func

        """

        """
        # Step 1: Only process the main flame front, ignore any small detatched flame kernels
        if len(flame_contours) > 1:
            flame_contour = max(flame_contours, key=len)
        else:
            flame_contour = flame_contours[0]

        # Step 1:
        data = raw_data.covering_grid(raw_data.max_level,
                                      left_edge=[0.0, 0.0, 0.0],
                                      dims=raw_data.domain_dimensions * [2 ** raw_data.max_level,
                                                                         2 ** raw_data.max_level, 1],
                                      # And any fields to preload (this is optional!)
                                      # fields=desired_varables
                                      )

        # Step 1: Separate the x and y components from the flame position array
        try:
            x_arr = np.array(flame_contour[:, 0], dtype=float)
            y_arr = np.array(flame_contour[:, 1], dtype=float)
        except:
            print('Error in Size Matching')

        # Step 2: Segment the flame contour based on multivaluedness (change in sign of dx)
        flame_x_segments = segment_flame_contour(x_arr, y_arr, plt_check=True)
        flame_y_segments = segment_flame_contour(y_arr, x_arr, plt_check=True)

        # Step 3:
        amplitudes, _, poly_fit = contour_poly_fit_and_distance(y_arr, x_arr, degree=9, plt_check=True)
        # Step 4:
        x_poly_fit_data = poly_fit(np.linspace(0, data.RightEdge.to_value()[1], len(y_arr)))
        poly_fit_power_spectrum = np.abs(np.fft.fft(x_poly_fit_data)) ** 2
        poly_fit_frequencies = np.fft.fftfreq(len(x_poly_fit_data),
                                              d=x_poly_fit_data[1] - x_poly_fit_data[0])  # Frequency bins

        # Step 5:
        amplitude_power_spectrum = np.abs(np.fft.fft(amplitudes)) ** 2
        amplitude_frequencies = np.fft.fftfreq(len(y_arr), d=data.dds.to_value()[1])  # Frequency bins

        plt.figure(figsize=(12, 10))
        # Plot the original data and wavelet reconstruction
        plt.subplot(2, 1, 1)
        plt.plot(y_arr, x_arr, label='Original')
        plt.plot(np.linspace(0, data.RightEdge.to_value()[1], len(y_arr)),
                 poly_fit(np.linspace(0, data.RightEdge.to_value()[1], len(y_arr)))
                 , label='Reconstruction', linestyle='dashed')
        plt.legend()
        plt.grid()
        #
        plt.subplot(2, 2, 3)
        plt.plot(amplitude_frequencies[:len(amplitude_frequencies) // 2],
                 amplitude_power_spectrum[:len(amplitude_frequencies) // 2],
                 label='Amplitude Spectrum')
        plt.xscale('log')
        plt.legend()
        plt.grid()
        #
        plt.subplot(2, 2, 4)
        plt.plot(poly_fit_frequencies[:len(poly_fit_frequencies) // 2],
                 poly_fit_power_spectrum[:len(poly_fit_frequencies) // 2],
                 label='Poly-Fit Spectrum')
        plt.xscale('log')
        plt.legend()
        plt.grid()

        plt.tight_layout()

        output_dir_path = ensure_long_path_prefix(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                               f"Processed-Global-Results",
                                                               f"Flame-Contour-FFT-Plt-Files"))
        if os.path.exists(output_dir_path) is False:
            os.mkdir(output_dir_path)

        formatted_time = '{:.16f}'.format(raw_data.current_time.to_value()).rstrip('0').rstrip('.')
        filename = os.path.join(output_dir_path, f"Flame-Countour-FFT-Time-{formatted_time}.png")
        plt.savefig(filename, format='png')
        plt.show()
        plt.close()

        return

    """

    :return: flame length for 2D simulation data
    """
    # Step 1: Load the current plt file
    raw_data.force_periodicity()
    all_data = raw_data.all_data()
    verts = all_data.extract_isocontours("Temp", 2000)
    rough_index = np.lexsort((verts[:, 0], verts[:, 1]))
    rough_sort = verts[rough_index]

    # Step 2: Remove outliers, artifacts of the periodicity, and order the array
    # Let us determine the buffer region caused by the periodic boundaries required
    buffer = 0.0125 * raw_data.domain_right_edge.to_value()[1]
    y_lower = raw_data.domain_left_edge.to_value()[1] + buffer
    y_upper = raw_data.domain_right_edge.to_value()[1] - buffer
    # Handling 2D contours from extracted vertices
    # Assuming verts contains 3D coordinates (x, y, z), we'll extract the 2D data
    # verts is a list of arrays, each array contains (x, y, z) vertices
    contour_pts = np.empty((len(rough_sort), 2), dtype=object)
    for i, vert in enumerate(rough_sort):
        # Extract only x and y for 2D visualization for valid points
        if (not np.isclose(vert[0], 0, atol=1e-02)
                and not np.isclose(vert[0], raw_data.domain_right_edge[0].to_value(), atol=1e-04)
                and (y_lower <= vert[1] <= y_upper)):
            if i == 0:
                contour_pts[i, 0] = vert[0]
                contour_pts[i, 1] = vert[1]
            if i > 0:
                if (vert[0] - rough_sort[i - 1, 0]) != 0 and (vert[1] - rough_sort[i - 1, 1]) != 0:
                    contour_pts[i, 0] = vert[0]
                    contour_pts[i, 1] = vert[1]

    contour_pts = np.array([x for x in contour_pts if x[0] is not None])
    #
    contour_arr = sort_by_nearest_neighbors(contour_pts)
    # Calculate the length of a line between adjacent points for the determine contour
    contour_segments = []
    current_contour = [contour_arr[0]]  # Start with the first point

    contour_lines = []
    current_line = []

    for i in range(1, len(contour_arr)):
        temp_var = np.sqrt((contour_arr[i, 0] - contour_arr[i - 1, 0]) ** 2 +
                           (contour_arr[i, 1] - contour_arr[i - 1, 1]) ** 2)

        if temp_var > 0.1 * raw_data.domain_right_edge[1].to_value():
            # If the distance exceeds the threshold, start a new contour line
            contour_segments.append(np.array(current_contour))
            current_contour = [contour_arr[i]]  # Start new contour with the current point

            contour_lines.append(np.array(current_line))
            current_line = []
        else:
            current_contour.append(contour_arr[i])
            current_line.append(temp_var)

    # Append the last contour line if not empty
    if current_contour:
        contour_segments.append(np.array(current_contour))
        contour_lines.append(np.array(current_line))

    # Determine the wavelength spectrum for a given flame contour using FFT
    # flame_front_wavelength_spectrum(raw_data, contour_segments)

    # Calculate the total line length, by summing the lines between points
    surface_length = sum(np.sum(distances) for distances in contour_lines)
    # print('Flame Surface Length: ', surface_length, 'cm')

    return surface_length / 100


def cantera_ignition_function(plt_data, ddt_window, processing_flags, input_params, reactor_type='Pressure'):
    ####################################################################################################################
    # This function uses cantera to compute the ignition delay for a given mixture at a given initial state
    ####################################################################################################################
    reactor_arr = np.empty(processing_flags['Processing Parameters']['Spatial Window'], dtype=object)

    spatial_idx = np.arange(ddt_window[0][0], ddt_window[1][0] + 1)
    for i, value in enumerate(spatial_idx):
        # Step 1: Collect the initial conditions over the physical window
        species_comp = {}
        for j in range(len(input_params.species)):
            species_comp.update({str(input_params.species[j]):
                                     plt_data["boxlib", str("Y(" + input_params.species[j] + ")")][
                                         value, ddt_window[0][1]].to_value().flatten()})

        # Step 2: Define the gas object and the initial state
        gas = ct.Solution(input_params.mech)
        gas.TPY = (plt_data["boxlib", "Temp"][value, ddt_window[0][1]].to_value().flatten(),
                   10 * plt_data["boxlib", 'pressure'][value, ddt_window[0][1]].to_value().flatten(),
                   species_comp)

        # Step 3:
        if reactor_type == 'Volume':
            r = ct.Reactor(contents=gas)

            gas_air = ct.Solution('air.yaml')
            gas_air.TPX = 300, 1.0 * ct.one_atm, {'N2': 0.79, 'O2': 0.21}
            env = ct.Reservoir(gas_air)
            w = ct.Wall(r, env)

            reactorNetwork = ct.ReactorNet([r])

        else:
            r = ct.IdealGasConstPressureReactor(contents=gas)
            reactorNetwork = ct.ReactorNet([r])

        # Step 4:
        timeHistory = ct.SolutionArray(gas, extra=['t'])

        # Step 4: Run the reactor and determine the time history of the reaction occuring
        t = 0
        tFinal = 600
        while t < tFinal:
            # Step 4.1: Take a time step and determine the new state of the reactor
            t = reactorNetwork.step()
            # Step 4.2: Save the new state of the reactor to the timeHistory object
            timeHistory.append(r.thermo.state, t=t)

        # Step 4:
        result_dict = {}
        result_dict['Time'] = timeHistory.t
        if processing_flags['Ignition Delay Processing'].get('Temperature', False):
            result_dict['Temperature'] = timeHistory.T
        if processing_flags['Ignition Delay Processing'].get('Pressure', False):
            result_dict['Pressure'] = timeHistory.P
        if processing_flags['Ignition Delay Processing'].get('Density', False):
            result_dict['Density'] = timeHistory.density_mass
        if processing_flags['Ignition Delay Processing'].get('Species', False):
            result_dict['Species'] = timeHistory.Y
        if processing_flags['Ignition Delay Processing'].get('Heat Release Rate', False):
            result_dict['Heat Release Rate'] = timeHistory.heat_release_rate

        # Step 5: Create a dictionary containing the results
        reactor_arr[i] = result_dict

        del r, reactorNetwork

        # Step 6:
        plt_check = False
        if plt_check:
            plt.figure(figsize=(8, 6))
            plt.plot(timeHistory.t, timeHistory.T, linestyle='-', color='k')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.show()

    return reactor_arr


def cantera_flame_function(plt_data, ddt_window, processing_flags, input_params):
    flame_arr = np.empty(processing_flags['Processing Parameters']['Spatial Window'], dtype=object)

    spatial_idx = np.arange(ddt_window[0][0], ddt_window[1][0] + 1)
    for i, value in enumerate(spatial_idx):
        # Step 1: Collect the initial conditions over the physical window
        species_comp = {}
        for j in range(len(input_params.species)):
            species_comp.update({str(input_params.species[j]):
                                     plt_data["boxlib", str("Y(" + input_params.species[j] + ")")][
                                         value, ddt_window[0][1]].to_value().flatten()})

        # Step 2: Define the gas object and the initial state
        gas = ct.Solution(input_params.mech)
        gas.TPY = (plt_data["boxlib", 'Temp'][value, ddt_window[0][1]].to_value().flatten(),
                   10 * plt_data["boxlib", 'pressure'][value, ddt_window[0][1]].to_value().flatten(),
                   species_comp)
        # Step 3: Simulation parameters
        width = 0.0001  # m
        loglevel = 0  # amount of diagnostic output (0 to 8)
        # Step 4: Set up flame object
        f = ct.FreeFlame(gas, width=width)
        f.set_refine_criteria(ratio=3, slope=0.01, curve=0.01)
        f.show()
        # Step 5: Solve with mixture-averaged transport model
        f.transport_model = 'mixture-averaged'
        f.solve(loglevel=loglevel, auto=True)
        # Step 6:
        result_dict = {}
        result_dict['Grid'] = f.grid
        if processing_flags['Flame Processing'].get('Temperature', False):
            result_dict['Temperature'] = f.T
        if processing_flags['Flame Processing'].get('Pressure', False):
            result_dict['Pressure'] = f.P
        if processing_flags['Flame Processing'].get('Density', False):
            result_dict['Density'] = f.density_mass
        if processing_flags['Flame Processing'].get('Species', False):
            result_dict['Species'] = f.Y
        if processing_flags['Flame Processing'].get('Velocity', False):
            result_dict['Velocity'] = f.velocity
        if processing_flags['Flame Processing'].get('Heat Release Rate', False):
            result_dict['Heat Release Rate'] = f.heat_release_rate

        # Step 5: Create a dictionary containing the results
        flame_arr[i] = result_dict

        # Step 6:
        plt_check = False
        if plt_check:
            plt.figure(figsize=(8, 6))
            plt.plot(f.grid, f.T, linestyle='-', color='k')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.show()

        del gas, f

    return flame_arr


def createVariablePltFrame(raw_data, sort_arr, time, min_bounds, max_bounds, tracking_obj, domain_info, hrr_arr=None):
    def pltAnimationFrame():
        """
            Plots the data and saves the plot as an image file.

            Parameters:
            - x: array-like, the data for the x-axis.
            - y: array-like, the data for the y-axis.
            - filename: str, the name of the file to save the plot.
            - xlabel: str, label for the x-axis (default is 'X-axis').
            - ylabel: str, label for the y-axis (default is 'Y-axis').
            - title: str, title of the plot (default is 'Plot').
            - format: str, the format to save the plot in (default is 'png').
            """
        line_styles = ['-', '--', '-.', ':']
        style_cycle = itertools.cycle(line_styles)

        fig, ax1 = plt.subplots()
        if isinstance(tracking_obj, (np.ndarray, list)):
            ax1.plot(raw_data["boxlib", str('x')][sort_arr].to_value(),
                     raw_data["boxlib", str(tracking_obj[0])][sort_arr].to_value(),
                     label=tracking_obj[0], linestyle='-', color='k')
            ax1.set_xlabel('Position [cm]')
            ax1.set_ylabel(str(tracking_obj[0]))
            ax1.grid(True, axis='x')  # Only x-axis grid lines
            ax1.set_ylim(y_limit_min[0], y_limit_max[0])

            # Create a second Y-axis
            ax2 = ax1.twinx()
            ax2.plot(raw_data["boxlib", str('x')][sort_arr].to_value(),
                     raw_data["boxlib", str(tracking_obj[1])][sort_arr].to_value(),
                     label=tracking_obj[1], linestyle='--', color='r')
            if tracking_obj[1] == "pressure":
                plt.yscale('log')
            ax2.set_ylabel(str(tracking_obj[1]))
            ax2.set_ylim(y_limit_min[1], y_limit_max[1])

            wrapped_title = "\n".join(textwrap.wrap(
                f"{tracking_obj[0]} and {tracking_obj[1]} Variation at y = {domain_size[1]} and t = {time}", width=55))
        else:
            if tracking_obj == 'HRR':
                ax1.plot(raw_data["boxlib", str('x')][sort_arr].to_value(), hrr_arr,
                         label=tracking_obj, linestyle='-', color='k')
            else:
                ax1.plot(raw_data["boxlib", str('x')][sort_arr].to_value(),
                         raw_data["boxlib", str(tracking_obj)][sort_arr].to_value(),
                         label=tracking_obj, linestyle='-', color='k')
                if tracking_obj == "pressure":
                    plt.yscale('log')

            ax1.set_ylabel(str(tracking_obj))
            ax1.set_ylim(y_limit_min, y_limit_max)

            wrapped_title = "\n".join(
                textwrap.wrap(f"{tracking_obj} Variation at y = {domain_size[1]} and t = {time}", width=55))

        plt.suptitle(wrapped_title, ha="center")
        plt.xlim(0, domain_size[0])
        plt.legend()

        # Convert the float to a string without scientific notation
        formatted_time = '{:.16f}'.format(time).rstrip('0').rstrip('.')
        # Save the plot to a file
        if isinstance(tracking_obj, (np.ndarray, list)):
            filename = os.path.join(output_dir_path,
                                    f"{tracking_obj[0]}-{tracking_obj[1]}-Animation-Time-{formatted_time}.png")
        else:
            filename = os.path.join(output_dir_path, f"{tracking_obj}-Animation-Time-{formatted_time}.png")
        plt.savefig(filename, format='png')
        plt.close()  # Close the figure to avoid displaying it inline if using in a Jupyter notebook
        return

    """

    """
    # Step 1:
    if isinstance(tracking_obj, (np.ndarray, list)):
        y_limit_min = np.zeros(len(tracking_obj))
        y_limit_max = np.zeros(len(tracking_obj))
        for i in range(len(tracking_obj)):
            y_limit_min[i] = min_bounds[i]
            y_limit_max[i] = max_bounds[i] + (0.1 * max_bounds[i])
    else:
        y_limit_min = min_bounds
        y_limit_max = max_bounds + (0.1 * max_bounds)

    # Step 2:
    domain_size = domain_info[0][1]
    # Step 3: Check/Create directory to store animation frmaes
    if isinstance(tracking_obj, (np.ndarray, list)):
        output_dir_path = ensure_long_path_prefix(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                               f"Processed-Global-Results", f"Animation-Frames",
                                                               f"{tracking_obj[0]}-{tracking_obj[1]}-Plt-Files"))
    else:
        output_dir_path = ensure_long_path_prefix(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                               f"Processed-Global-Results", f"Animation-Frames",
                                                               f"{tracking_obj}-Plt-Files"))

    # Step 3:
    pltAnimationFrame()

    return


def createVariableAnimation(folder_path, output_filename, fps=15):
    def collect_plot_files(folder_path, file_extension='png'):
        """
        Collects all plot files from a given folder.

        Parameters:
        - folder_path: str, the path to the folder containing plot files.
        - file_extension: str, the file extension of the plot files (default is 'png').

        Returns:
        - image_files: list of str, paths to the image files sorted by filename.
        """
        # List all files in the directory with the given extension
        image_files = [os.path.join(folder_path, f) for f in sorted(os.listdir(folder_path)) if
                       f.endswith(f'.{file_extension}')]
        return image_files

    def create_animation(image_files, output_filename, fps=2):
        """
        Creates an MP4 animation from a series of image files.

        Parameters:
        - image_files: list of str, paths to the image files.
        - output_filename: str, the name of the output video file (e.g., 'animation.mp4').
        - fps: int, frames per second for the animation (default is 2).
        """
        # Create a figure and axis
        fig, ax = plt.subplots()

        # Load the first image to get the size
        first_image = mpimg.imread(image_files[0])
        ax.set_xlim(0, first_image.shape[1])
        ax.set_ylim(0, first_image.shape[0])

        # Placeholder for the image
        img_display = ax.imshow(first_image)

        # Update function for animation
        def update(frame):
            img_display.set_array(np.flipud(mpimg.imread(image_files[frame])))
            return img_display,

        # Create animation
        ani = animation.FuncAnimation(fig, update, frames=len(image_files), blit=True)

        # Choose the writer and save the animation
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=fps, metadata=dict(artist='Me'), bitrate=1800)

        # Save the animation as MP4
        ani.save(output_filename, writer=writer)

        plt.close()  # Close the plot

    """
        Creates an MP4 animation from plot files in a given folder.

        Parameters:
        - folder_path: str, the path to the folder containing plot files.
        - output_filename: str, the name of the output video file (e.g., 'animation.mp4').
        - fps: int, frames per second for the animation (default is 2).
        - file_extension: str, the file extension of the plot files (default is 'png').
    """
    # Collect all plot files from the folder
    image_files = collect_plot_files(folder_path, 'png')

    # Create animation using the collected plot files
    create_animation(image_files, output_filename, fps)
    return


def single_pltfile_processing(args):
    def load_data():
        raw_data = yt.load(pltFile_dir)
        slice = raw_data.ray(np.array([0.0, domain_size[1], 0.0]), np.array([domain_size[0], domain_size[1], 0.0]))
        ray_sort = np.argsort(slice["x"])
        time = raw_data.current_time.to_value()

        plt_check = False
        if plt_check:
            plt.figure(figsize=(8, 6))
            plt.plot(slice['x'][ray_sort], slice['Temp'][ray_sort], linestyle='-', color='k')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.show()

        return time, raw_data, slice, ray_sort

    """

    """
    # Step 1: Unpack the argumnents provided
    pltFile_dir = args[0]
    processing_flags = args[1][0]
    animation_pltfiles = args[1][1]
    domain_info = args[1][2]
    domain_size = domain_info[0][1]
    input_params = args[1][3]

    # Step 2: Load the pltFile for individual processing
    time, raw_data, plt_data, sort_arr = load_data()

    # Step 3:
    result_dict = {}
    result_dict['Time'] = {}
    result_dict['Time']['Value'] = time

    # Step 3.1: Flame Processing
    if 'Flame Processing' in processing_flags:
        result_dict['Flame'] = {}

        # Position
        if processing_flags['Flame Processing'].get('Position', False):
            [result_dict['Flame']['Index'], result_dict['Flame']['Position']] = wave_tracking_function(plt_data,
                                                                                                       sort_arr, "Temp",
                                                                                                       "Flame")

        # Thermodynamic State
        if processing_flags['Flame Processing'].get('Thermodynamic State', False):
            result_dict['Flame']['Thermodynamic State'] = thermodynamic_state_function(plt_data, sort_arr,
                                                                                       result_dict['Flame']['Index'],
                                                                                       "Flame", input_params)

        # Flame Heat Release Rate
        if processing_flags['Flame Processing'].get('Heat Release Rate', False):
            [result_dict['Flame']['Heat Release Rate'],
             result_dict['Flame']['Max Heat Release Rate']] = heat_release_rate_function(plt_data, sort_arr,
                                                                                         input_params)

        # Surface Flame Thickness
        if processing_flags['Flame Processing'].get('Flame Thickness', False):
            result_dict['Flame']['Flame Thickness'] = flame_thickness_function(plt_data, sort_arr,
                                                                               result_dict['Flame']['Index'])

        # Surface Length
        if processing_flags['Flame Processing'].get('Surface Length', False):
            try:
                result_dict['Flame']['Surface Length'] = flame_length_contour_function(raw_data, plt_data, sort_arr)
            except:
                result_dict['Flame']['Surface Length'] = 0

    # Step 3.2: Maximum Pressure Processing
    if 'Maximum Pressure Processing' in processing_flags:
        result_dict['Maximum Pressure'] = {}

        # Position
        if processing_flags['Maximum Pressure Processing'].get('Position', False):
            [result_dict['Max Pressure']['Index'], result_dict['Max Pressure']['Position']] = wave_tracking_function(
                plt_data, sort_arr, "presssure", "Maximum Pressure")

        # Thermodynamic State
        if processing_flags['Maximum Pressure Processing'].get('Thermodynamic State', False):
            result_dict['Max Pressure']['Thermodynamic State'] = thermodynamic_state_function(plt_data, sort_arr,
                                                                                              result_dict[
                                                                                                  'Max Pressure'][
                                                                                                  'Index'],
                                                                                              "Maximum Pressure",
                                                                                              input_params)

    # Step 3.2: Leading Shock Wave Processing
    if 'Leading Shock Processing' in processing_flags:
        result_dict['Lead Shock'] = {}

        # Position
        if processing_flags['Leading Shock Processing'].get('Position', False):
            [result_dict['Lead Shock']['Index'], result_dict['Lead Shock']['Position']] = wave_tracking_function(
                plt_data, sort_arr, "pressure", "Leading Shock")

    # Step 3.3: Pre-Shock Processing
    if 'Pre-Shock Processing' in processing_flags:
        result_dict['Pre-Shock'] = {}

        if processing_flags['Pre-Shock Processing'].get('Thermodynamic State', False):
            # If the leading shock wave is not flagged, determine the location here
            if not processing_flags['Leading Shock Processing'].get('Position', False):
                result_dict['Lead Shock'] = {}
                [result_dict['Lead Shock']['Index'], _] = wave_tracking_function(plt_data, sort_arr, "pressure",
                                                                                 "Leading Shock")

            # Determine the thermodynamic state ahead of the leading shock wave
            result_dict['Pre-Shock']['Thermodynamic State'] = thermodynamic_state_function(plt_data, sort_arr,
                                                                                           result_dict['Lead Shock'][
                                                                                               'Index'], "Pre-Shock",
                                                                                           input_params)

    # Step 3.4: Post-Shock Processing
    if 'Post-Shock Processing' in processing_flags:
        result_dict['Post-Shock'] = {}

        if processing_flags['Post-Shock Processing'].get('Thermodynamic State', False):
            # If the leading shock wave is not flagged, determine the location here
            if not processing_flags['Leading Shock Processing'].get('Position', False):
                result_dict['Lead Shock'] = {}
                [result_dict['Lead Shock']['Index'], _] = wave_tracking_function(plt_data, sort_arr, "pressure",
                                                                                 "Leading Shock")

            # Determine the thermodynamic state behind of the leading shock wave
            result_dict['Post-Shock']['Thermodynamic State'] = thermodynamic_state_function(plt_data, sort_arr,
                                                                                            result_dict['Lead Shock'][
                                                                                                'Index'], "Post-Shock",
                                                                                            input_params)

    # Step 3.5: State Variation Animation
    if 'Domain State Animations' in processing_flags:
        # Step 3.5.1: Get first time step to determine minimum parameters
        temp_data = yt.load(animation_pltfiles[0])
        temp_slice = temp_data.ray(np.array([0.0, domain_info[0][1][1], 0.0]),
                                   np.array([domain_info[0][1][0], domain_info[0][1][1], 0.0]))

        min_val_marker = np.empty(4 + len(input_params.species), dtype=object)
        min_val_arr = np.zeros(4 + len(input_params.species))
        for i in range(len(min_val_arr)):
            if i == 0:
                min_val_marker[i] = str('Temp')
                min_val_arr[i] = np.min(temp_slice["boxlib", str('Temp')])
            elif i == 1:
                min_val_marker[i] = 'pressure'
                min_val_arr[i] = np.min(temp_slice["boxlib", str('pressure')])
            elif i == 2:
                min_val_marker[i] = 'x_velocity'
                min_val_arr[i] = np.min(temp_slice["boxlib", str('x_velocity')])
            elif i == 3:
                min_val_marker[i] = 'HRR'
                min_val_arr[i] = np.min(result_dict['Flame']['Heat Release Rate'])
            else:
                min_val_marker[i] = str("Y(" + input_params.species[i - 4] + ")")
                min_val_arr[i] = np.min(temp_slice["boxlib", str("Y(" + input_params.species[i - 4] + ")")])

        # Step 3.5.2: Get last time step to determine minimum parameters
        temp_data = yt.load(animation_pltfiles[1])
        temp_slice = temp_data.ray(np.array([0.0, domain_info[0][1][1], 0.0]),
                                   np.array([domain_info[0][1][0], domain_info[0][1][1], 0.0]))

        max_val_marker = np.empty(4 + len(input_params.species), dtype=object)
        max_val_arr = np.zeros(4 + len(input_params.species))
        for i in range(len(max_val_arr)):
            if i == 0:
                max_val_marker[i] = str('Temp')
                max_val_arr[i] = np.max(temp_slice["boxlib", str('Temp')])
            elif i == 1:
                max_val_marker[i] = 'pressure'
                max_val_arr[i] = np.max(temp_slice["boxlib", str('pressure')])
            elif i == 2:
                max_val_marker[i] = 'x_velocity'
                max_val_arr[i] = np.max(temp_slice["boxlib", str('x_velocity')])
            elif i == 3:
                max_val_marker[i] = 'HRR'
                max_val_arr[i] = np.max(result_dict['Flame']['Heat Release Rate'])
            else:
                max_val_marker[i] = str("Y(" + input_params.species[i - 4] + ")")
                max_val_arr[i] = np.max(temp_slice["boxlib", str("Y(" + input_params.species[i - 4] + ")")])

        # Step 3.5.3: Plot Temperature
        if processing_flags['Domain State Animations'].get('Temperature', False):
            # Create directory for plt files
            temp_plt_dir = ensure_long_path_prefix(
                os.path.join(os.path.dirname(os.path.realpath(__file__)), f"Processed-Global-Results",
                             f"Animation-Frames", f"Temp-Plt-Files"))

            if os.path.exists(temp_plt_dir) is False:
                os.makedirs(temp_plt_dir, exist_ok=True)

            createVariablePltFrame(plt_data, sort_arr, time, min_val_arr[0], max_val_arr[0], "Temp", domain_info)

        # Step 3.5.4: Plot Pressure
        if processing_flags['Domain State Animations'].get('Pressure', False):
            # Create directory for plt files
            temp_plt_dir = ensure_long_path_prefix(
                os.path.join(os.path.dirname(os.path.realpath(__file__)), f"Processed-Global-Results",
                             f"Animation-Frames", f"pressure-Plt-Files"))

            if os.path.exists(temp_plt_dir) is False:
                os.makedirs(temp_plt_dir, exist_ok=True)

            createVariablePltFrame(plt_data, sort_arr, time, min_val_arr[1], max_val_arr[1], "pressure", domain_info)

        # Step 3.5.5: Plot Velocity
        if processing_flags['Domain State Animations'].get('Velocity', False):
            # Create directory for plt files
            temp_plt_dir = ensure_long_path_prefix(
                os.path.join(os.path.dirname(os.path.realpath(__file__)), f"Processed-Global-Results",
                             f"Animation-Frames", f"x_velocity-Plt-Files"))

            if os.path.exists(temp_plt_dir) is False:
                os.makedirs(temp_plt_dir, exist_ok=True)

            createVariablePltFrame(plt_data, sort_arr, time, min_val_arr[2], max_val_arr[2], "x_velocity", domain_info)

        # Step 3.5.5: Plot Velocity
        if processing_flags['Domain State Animations'].get('Heat Release Rate', False):
            # Create directory for plt files
            temp_plt_dir = ensure_long_path_prefix(
                os.path.join(os.path.dirname(os.path.realpath(__file__)), f"Processed-Global-Results",
                             f"Animation-Frames", f"HRR-Plt-Files"))

            if os.path.exists(temp_plt_dir) is False:
                os.makedirs(temp_plt_dir, exist_ok=True)

            createVariablePltFrame(plt_data, sort_arr, time, min_val_arr[3], max_val_arr[3], "HRR",
                                   domain_info, hrr_arr=result_dict['Flame']['Heat Release Rate'])

        # Step 3.5.6: Plot Species
        if processing_flags['Domain State Animations'].get('Species', False):
            for i in range(len(input_params.species)):
                # Create directory for plt files
                temp_plt_dir = ensure_long_path_prefix(
                    os.path.join(os.path.dirname(os.path.realpath(__file__)), f"Processed-Global-Results",
                                 f"Animation-Frames", f"Y({str(input_params.species[i])})-Plt-Files"))
                if os.path.exists(temp_plt_dir) is False:
                    os.makedirs(temp_plt_dir, exist_ok=True)

                createVariablePltFrame(plt_data, sort_arr, time, min_val_arr[i + 4], max_val_arr[i + 4],
                                       f"Y({str(input_params.species[i])})", domain_info)

        # Step 3.5.3: Plot Combined
        if processing_flags['Domain State Animations'].get('Combined', False):
            # Create directory for plt files
            temp_plt_dir = ensure_long_path_prefix(
                os.path.join(os.path.dirname(os.path.realpath(__file__)), f"Processed-Global-Results",
                             f"Animation-Frames", f"{processing_flags['Domain State Animations']['Combined'][0]}-"
                                                  f"{processing_flags['Domain State Animations']['Combined'][1]}-Plt-Files"))
            if os.path.exists(temp_plt_dir) is False:
                os.makedirs(temp_plt_dir, exist_ok=True)

            createVariablePltFrame(plt_data, sort_arr, time,
                                   [min_val_arr[np.argwhere(
                                       min_val_marker == processing_flags['Domain State Animations']['Combined'][0])][
                                        0][0],
                                    min_val_arr[np.argwhere(
                                        min_val_marker == processing_flags['Domain State Animations']['Combined'][1])][
                                        0][0]],
                                   [max_val_arr[np.argwhere(
                                       max_val_marker == processing_flags['Domain State Animations']['Combined'][0])][
                                        0][0],
                                    max_val_arr[np.argwhere(
                                        max_val_marker == processing_flags['Domain State Animations']['Combined'][1])][
                                        0][0]],
                                   [processing_flags['Domain State Animations']['Combined'][0],
                                    processing_flags['Domain State Animations']['Combined'][1]], domain_info)

    return result_dict


def single_ddt_pltfile_processing(args):
    def load_data():
        raw_data = yt.load(pltFile_dir)
        max_level = raw_data.index.max_level

        # Create sudo-grid at the maximum level present
        temp_ddt_data = raw_data.covering_grid(level=max_level,
                                               left_edge=[0.0, 0.0, 0.0],
                                               dims=raw_data.domain_dimensions * [2 ** max_level, 2 ** max_level, 1],
                                               # And any fields to preload (this is optional!)
                                               # fields=desired_varables
                                               )

        time = raw_data.current_time.to_value()

        plt_check = False
        if plt_check:
            plt.figure(figsize=(8, 6))
            plt.plot(temp_ddt_data["boxlib", "x"][:, 0].to_value().flatten(),
                     temp_ddt_data["boxlib", "Temp"][:, 0].to_value().flatten(), linestyle='-', color='k')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.show()

        return time, raw_data, temp_ddt_data

    """

    """
    # Step 1: Unpack the argumnents provided
    pltFile_dir = args[0]
    ddt_window = args[1][0]
    processing_flags = args[1][1]
    animation_pltfiles = args[1][2]
    input_params = args[1][3]

    spatial_idx = np.arange(ddt_window[0][0], ddt_window[1][0] + 1)

    # Step 3: Load the pltFile for individual processing
    time, raw_data, plt_data = load_data()

    # Step 4:
    result_dict = {}
    result_dict['Time'] = time

    # Step 4.1: Collect the PeleC Domain Information
    result_dict['PeleC'] = {}
    result_dict['PeleC']['Temperature'] = np.empty(len(spatial_idx))
    result_dict['PeleC']['Pressure'] = np.empty(len(spatial_idx))
    result_dict['PeleC']['Density'] = np.empty(len(spatial_idx))
    result_dict['PeleC']['Species'] = np.empty(len(spatial_idx), dtype=object)
    result_dict['PeleC']['Heat Release Rate'] = np.empty(len(spatial_idx))
    result_dict['PeleC']['Max Heat Release Rate'] = np.empty(len(spatial_idx))

    for i, value in enumerate(spatial_idx):
        if processing_flags['PeleC Processing'].get('Temperature', False):
            result_dict['PeleC']['Temperature'][i] = plt_data["boxlib", 'Temp'][value, ddt_window[0][1]][0].to_value()
        if processing_flags['PeleC Processing'].get('Pressure', False):
            result_dict['PeleC']['Pressure'][i] = 10 * plt_data["boxlib", 'pressure'][value, ddt_window[0][1]][
                0].to_value()
        if processing_flags['PeleC Processing'].get('Density', False):
            result_dict['PeleC']['Density'][i] = 1000 * plt_data["boxlib", 'density'][value, ddt_window[0][1]][
                0].to_value()

        if processing_flags['PeleC Processing'].get('Species', False):
            species_comp = {}
            for j in range(len(input_params.species)):
                species_comp.update({str(input_params.species[j]):
                                         plt_data["boxlib", str("Y(" + input_params.species[j] + ")")][
                                             value, ddt_window[0][1]][0].to_value()})
            result_dict['PeleC']['Species'][i] = species_comp

        if processing_flags['PeleC Processing'].get('Heat Release Rate', False):
            [result_dict['PeleC']['Heat Release Rate'][i],
             result_dict['PeleC']['Max Heat Release Rate'][i]] = heat_release_rate_function(plt_data, spatial_idx,
                                                                                            input_params)

        # Step 4.2: Constant Pressure Ignition Delay Processing
    if processing_flags['Comparison Type'].get('Ignition Delay Pressure', False):
        # Step 4.2.1: Calculate the ignition behavior for a given initial state
        result_dict['Ignition Delay Pressure'] = cantera_ignition_function(plt_data, ddt_window, processing_flags,
                                                                           input_params)

    # Step 4.3: Constant Volume Ignition Delay Processing
    if processing_flags['Comparison Type'].get('Ignition Delay Volume', False):
        # Step 4.3.1: Calculate the ignition behavior for a given initial state
        result_dict['Ignition Delay Volume'] = cantera_ignition_function(plt_data, ddt_window, processing_flags,
                                                                         input_params, reactor_type='Volume')

    # Step 4.4: Flame Processing
    if processing_flags['Comparison Type'].get('Flame', False):
        # Step 4.4.1: Calculate the flame behavior for a given initial state
        result_dict['Flame'] = cantera_flame_function(plt_data, ddt_window, processing_flags, input_params)

    return result_dict


def pelec_processing_function(plt_dir, domain_info, input_params, check_flags, output_dir=None):
    def parallel_processing_function(iter_arr, const_list, predicate, nProcs):
        """

        :param iter_arr:
        :param const_list:
        :param predicate:
        :param nProcs:
        :return:
        """
        # Perform the multiprocessing
        with multiprocessing.Pool(
                processes=nProcs, initargs=()
        ) as pool:
            y = pool.map(predicate,
                         zip(iter_arr,
                             itertools.repeat(const_list)))
        return y

    def file_output(file_path, smoothing_check):
        # Step 1: Dynamically create the text file header depending on the assigned flags
        header_data = ["Time [s]"]
        # Position
        if check_flags['Flame Processing'].get('Position', False):
            header_data.extend(["Flame Position [m]"])
        if check_flags['Leading Shock Processing'].get('Position', False):
            header_data.extend(["Leading Shock Position [m]"])
        if check_flags['Maximum Pressure Processing'].get('Position', False):
            header_data.extend(["Max Pressure Position [m]"])
        # Velocity
        if check_flags['Flame Processing'].get('Velocity', False):
            header_data.extend(["Flame Velocity [m/s]"])
        if check_flags['Leading Shock Processing'].get('Velocity', False):
            header_data.extend(["Leading Shock Velocity [m/s]"])
        # Relative Velocity
        if check_flags['Flame Processing'].get('Relative Velocity', False):
            header_data.extend(["Relative Flame Velocity [m/s]"])
        # Thermodynamic State
        if check_flags['Flame Processing'].get('Thermodynamic State', False):
            header_data.extend(["Flame Temperature [K]", "Flame Pressure [Pa]", "Flame Density [kg/m^3]",
                                "Flame Soundspeed [m/s]"])
        if check_flags['Maximum Pressure Processing'].get('Thermodynamic State', False):
            header_data.extend(["Max Pressure Temperature [K]", "Max Pressure Pressure [Pa]",
                                "Max Pressure Density [kg/m^3]", "Max Pressure Soundspeed [m/s]"])
        if check_flags['Pre-Shock Processing'].get('Thermodynamic State', False):
            header_data.extend(["Pre-Shock Temperature [K]", "Pre-Shock Pressure [Pa]",
                                "Pre-Shock Density [kg/m^3]", "Pre-Shock Soundspeed [m/s]"])
        if check_flags['Post-Shock Processing'].get('Thermodynamic State', False):
            header_data.extend(["Post-Shock Temperature [K]", "Post-Shock Pressure [Pa]",
                                "Post-Shock Density [kg/m^3]", "Post-Shock Soundspeed [m/s]"])
        # Heat Release Rate
        if check_flags['Flame Processing'].get('Heat Release Rate', False):
            header_data.extend(["Max Heat Release Rate [W/m3]"])
        # Flame Thickness
        if check_flags['Flame Processing'].get('Flame Thickness', False):
            header_data.extend(["Flame Thickness [m]"])
        # Flame Surface Length
        if check_flags['Flame Processing'].get('Surface Length', False):
            header_data.extend(["Flame Surface Length [m]"])

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
                    # Time
                    outfile.write(" {0:<55e}".format(collective_results['Time']['Value'][i]))

                    # Position
                    if check_flags['Flame Processing'].get('Position', False):
                        outfile.write(" {0:<55e}".format(collective_results['Flame']['Position'][i]))

                    if check_flags['Maximum Pressure Processing'].get('Position', False):
                        outfile.write(" {0:<55e}".format(collective_results['Maximum Pressure']['Position'][i]))

                    if check_flags['Leading Shock Processing'].get('Position', False):
                        outfile.write(" {0:<55e}".format(collective_results['Lead Shock']['Position'][i]))

                    # Velocity
                    if check_flags['Flame Processing'].get('Velocity', False):
                        outfile.write(" {0:<55e}".format(collective_results['Flame']['Velocity'][i]))

                    if check_flags['Leading Shock Processing'].get('Velocity', False):
                        outfile.write(" {0:<55e}".format(collective_results['Lead Shock']['Velocity'][i]))

                    # Relative Velocity
                    if check_flags['Flame Processing'].get('Relative Velocity', False):
                        outfile.write(" {0:<55e}".format(collective_results['Flame']['Relative Velocity'][i]))

                    # Thermodynamic State
                    if check_flags['Flame Processing'].get('Thermodynamic State', False):
                        for j in range(len(collective_results['Flame']['Thermodynamic State'][i])):
                            outfile.write(" {0:<55e}".format(collective_results['Flame']['Thermodynamic State'][i][j]))

                    if check_flags['Maximum Pressure Processing'].get('Thermodynamic State', False):
                        for j in range(len(collective_results['Maximum Pressure']['Thermodynamic State'][i])):
                            outfile.write(
                                " {0:<55e}".format(collective_results['Maximum Pressure']['Thermodynamic State'][i][j]))

                    if check_flags['Pre-Shock Processing'].get('Thermodynamic State', False):
                        for j in range(len(collective_results['Pre-Shock']['Thermodynamic State'][i])):
                            outfile.write(
                                " {0:<55e}".format(collective_results['Pre-Shock']['Thermodynamic State'][i][j]))

                    if check_flags['Post-Shock Processing'].get('Thermodynamic State', False):
                        for j in range(len(collective_results['Post-Shock']['Thermodynamic State'][i])):
                            outfile.write(
                                " {0:<55e}".format(collective_results['Post-Shock']['Thermodynamic State'][i][j]))

                    # Heat Release Rate
                    if check_flags['Flame Processing'].get('Heat Release Rate', False):
                        outfile.write(" {0:<55e}".format(collective_results['Flame']['Max Heat Release Rate'][i]))

                    # Flame Thickness
                    if check_flags['Flame Processing'].get('Flame Thickness', False):
                        outfile.write(" {0:<55e}".format(collective_results['Flame']['Flame Thickness'][i]))

                    # Surface Length
                    if check_flags['Flame Processing'].get('Surface Length', False):
                        outfile.write(" {0:<55e}".format(collective_results['Flame']['Surface Length'][i]))
                    outfile.write("\n")

            if smoothing_check:
                for i in range(len(collective_results['Time']['Smooth']['Value'])):
                    # Time
                    outfile.write(" {0:<55e}".format(collective_results['Time']['Smooth']['Value'][i]))

                    # Position
                    if check_flags['Flame Processing'].get('Position', False):
                        outfile.write(" {0:<55e}".format(collective_results['Flame']['Smooth']['Position'][i]))

                    if check_flags['Maximum Pressure Processing'].get('Position', False):
                        outfile.write(
                            " {0:<55e}".format(collective_results['Maximum Pressure']['Smooth']['Position'][i]))

                    if check_flags['Leading Shock Processing'].get('Position', False):
                        outfile.write(" {0:<55e}".format(collective_results['Lead Shock']['Smooth']['Position'][i]))

                    # Velocity
                    if check_flags['Flame Processing'].get('Velocity', False):
                        outfile.write(" {0:<55e}".format(collective_results['Flame']['Smooth']['Velocity'][i]))

                    if check_flags['Leading Shock Processing'].get('Velocity', False):
                        outfile.write(" {0:<55e}".format(collective_results['Lead Shock']['Smooth']['Velocity'][i]))

                    # Relative Velocity
                    if check_flags['Flame Processing'].get('Relative Velocity', False):
                        outfile.write(" {0:<55e}".format(collective_results['Flame']['Smooth']['Relative Velocity'][i]))

                    # Thermodynamic State
                    if check_flags['Flame Processing'].get('Thermodynamic State', False):
                        for j in range(len(collective_results['Flame']['Smooth']['Thermodynamic State'])):
                            outfile.write(
                                " {0:<55e}".format(collective_results['Flame']['Smooth']['Thermodynamic State'][j][i]))

                    if check_flags['Maximum Pressure Processing'].get('Thermodynamic State', False):
                        for j in range(len(collective_results['Maximum Pressure']['Smooth']['Thermodynamic State'])):
                            outfile.write(
                                " {0:<55e}".format(
                                    collective_results['Maximum Pressure']['Smooth']['Thermodynamic State'][j][i]))

                    if check_flags['Pre-Shock Processing'].get('Thermodynamic State', False):
                        for j in range(len(collective_results['Pre-Shock']['Smooth']['Thermodynamic State'])):
                            outfile.write(
                                " {0:<55e}".format(
                                    collective_results['Pre-Shock']['Smooth']['Thermodynamic State'][j][i]))

                    if check_flags['Post-Shock Processing'].get('Thermodynamic State', False):
                        for j in range(len(collective_results['Post-Shock']['Smooth']['Thermodynamic State'])):
                            outfile.write(
                                " {0:<55e}".format(
                                    collective_results['Post-Shock']['Smooth']['Thermodynamic State'][j][i]))

                    # Heat Release Rate
                    if check_flags['Flame Processing'].get('Heat Release Rate', False):
                        outfile.write(
                            " {0:<55e}".format(collective_results['Flame']['Smooth']['Max Heat Release Rate'][i]))

                    # Flame Thickness
                    if check_flags['Flame Processing'].get('Flame Thickness', False):
                        outfile.write(" {0:<55e}".format(collective_results['Flame']['Smooth']['Flame Thickness'][i]))

                    # Surface Length
                    if check_flags['Flame Processing'].get('Surface Length', False):
                        outfile.write(" {0:<55e}".format(collective_results['Flame']['Smooth']['Surface Length'][i]))
                    outfile.write("\n")
            outfile.close()
            return

    """

    """
    # Step 1:
    print('Starting Raw Data Loading and Processing')
    plt_result = parallel_processing_function(plt_dir,
                                              (check_flags, [plt_dir[0], plt_dir[-1]], domain_info, input_params,),
                                              single_pltfile_processing, n_proc)
    print('Completed Raw Data Loading and Processing')

    # Step 2:
    collective_results = {}
    for master_key in plt_result[0]:
        if master_key not in collective_results:
            collective_results[master_key] = {}
        for slave_key in plt_result[0][master_key]:
            # Step 2.1
            temp_vec = []
            for i in range(len(plt_result)):
                temp_vec.append(plt_result[i][master_key][slave_key])
            # Step 2.2
            collective_results[master_key][slave_key] = temp_vec

    # Step 3:
    if 'Flame Processing' in check_flags:
        print('Starting Flame Processing')
        # Step 3.1: Velocity
        if check_flags['Flame Processing'].get('Velocity', False):
            collective_results['Flame']['Velocity'] = np.gradient(
                collective_results['Flame']['Position']) / np.gradient(collective_results['Time']['Value'])
        # Step 3.2: Relative Velocity
        if check_flags['Flame Processing'].get('Relative Velocity', False):
            print('ERROR: Relative Velocity is still a work in progress')
        # Step 3.3: Data Smoothing
        if check_flags['Flame Processing'].get('Smoothing', False) and check_flags['Flame Processing'].get('Position',
                                                                                                           False):
            smoothing_function(collective_results, 'Flame', 'Position')

        if check_flags['Flame Processing'].get('Smoothing', False) and check_flags['Flame Processing'].get('Velocity',
                                                                                                           False):
            smoothing_function(collective_results, 'Flame', 'Velocity')

        if check_flags['Flame Processing'].get('Smoothing', False) and check_flags['Flame Processing'].get(
                'Relative Velocity', False):
            smoothing_function(collective_results, 'Flame', 'Relative Velocity')

        if check_flags['Flame Processing'].get('Smoothing', False) and check_flags['Flame Processing'].get(
                'Thermodynamic State', False):
            smoothing_function(collective_results, 'Flame', 'Thermodynamic State')

        if check_flags['Flame Processing'].get('Smoothing', False) and check_flags['Flame Processing'].get(
                'Heat Release Rate', False):
            smoothing_function(collective_results, 'Flame', 'Max Heat Release Rate')

        if check_flags['Flame Processing'].get('Smoothing', False) and check_flags['Flame Processing'].get(
                'Flame Thickness', False):
            smoothing_function(collective_results, 'Flame', 'Flame Thickness')

        if check_flags['Flame Processing'].get('Smoothing', False) and check_flags['Flame Processing'].get(
                'Surface Length', False):
            smoothing_function(collective_results, 'Flame', 'Surface Length')

        print('Completed Flame Processing')
    # Step 4: Lead Shock Processing
    if 'Leading Shock Processing' in check_flags:
        print('Starting Lead Shock Processing')
        # Step 4.1: Velocity
        if check_flags['Leading Shock Processing'].get('Velocity', False):
            collective_results['Lead Shock']['Velocity'] = np.gradient(
                collective_results['Lead Shock']['Position']) / np.gradient(collective_results['Time']['Value'])
        # Step 4.1: Data Smoothing
        if check_flags['Leading Shock Processing'].get('Smoothing', False) and check_flags[
            'Leading Shock Processing'].get('Position', False):
            smoothing_function(collective_results, 'Lead Shock', 'Position')

        if check_flags['Leading Shock Processing'].get('Smoothing', False) and check_flags[
            'Leading Shock Processing'].get('Velocity', False):
            smoothing_function(collective_results, 'Lead Shock', 'Velocity')

        print('Completed Lead Shock Processing')
    # Step 5: Maximum Pressure Processing
    if 'Maximum Pressure Processing' in check_flags:
        print('Starting Max Pressure Processing')
        # Step 5.1: Data Smoothing
        if check_flags['Maximum Pressure Processing'].get('Smoothing', False) and check_flags[
            'Maximum Pressure Processing'].get('Position', False):
            smoothing_function(collective_results, 'Maximum Pressure', 'Position')

        if check_flags['Maximum Pressure Processing'].get('Smoothing', False) and check_flags[
            'Maximum Pressure Processing'].get('Thermodynamic State', False):
            smoothing_function(collective_results, 'Maximum Pressure', 'Thermodynamic State')

        print('Completed Max Pressure Processing')
    # Step 6: Pre-Shock Processing
    if 'Pre-Shock Processing' in check_flags:
        print('Starting Pre-Shock Processing')
        # Step 6.1: Data Smoothing
        if check_flags['Pre-Shock Processing'].get('Smoothing', False) and check_flags['Pre-Shock Processing'].get(
                'Thermodynamic State', False):
            smoothing_function(collective_results, 'Pre-Shock', 'Thermodynamic State')

        print('Completed Pre-Shock Processing')
    # Step 7: Post-Shock Processing
    if 'Post-Shock Processing' in check_flags:
        print('Starting Post-Shock Processing')
        # Step 7.1: Data Smoothing
        if check_flags['Post-Shock Processing'].get('Smoothing', False) and check_flags['Post-Shock Processing'].get(
                'Thermodynamic State', False):
            smoothing_function(collective_results, 'Post-Shock', 'Thermodynamic State')

        print('Completed Post-Shock Processing')
    # Step 7: Create Variable Evolution
    if 'Domain State Animations' in check_flags:
        print('Starting Animation Processing')
        # Step :
        animation_dir = ensure_long_path_prefix(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                             f"Processed-Global-Results"))

        if check_flags['Domain State Animations'].get('Flame Spectrum', False):
            # Create directory for plt files
            temp_plt_dir = ensure_long_path_prefix(
                os.path.join(os.path.dirname(os.path.realpath(__file__)), f"Processed-Global-Results",
                             f"Flame-Contour-FFT-Plt-Files"))

            # Create the animation file
            createVariableAnimation(temp_plt_dir,
                                    os.path.join(animation_dir, 'Flame-Frequency-Evolution-Animation.mp4'), fps=15)

        if check_flags['Domain State Animations'].get('Temperature', False):
            # Create directory for plt files
            temp_plt_dir = ensure_long_path_prefix(
                os.path.join(os.path.dirname(os.path.realpath(__file__)), f"Processed-Global-Results",
                             f"Animation-Frames", f"Temp-Plt-Files"))

            # Create the animation file
            createVariableAnimation(temp_plt_dir,
                                    os.path.join(animation_dir, 'Temperature-Evolution-Animation.mp4'), fps=15)

        if check_flags['Domain State Animations'].get('Pressure', False):
            # Create directory for plt files
            temp_plt_dir = ensure_long_path_prefix(
                os.path.join(os.path.dirname(os.path.realpath(__file__)), f"Processed-Global-Results",
                             f"Animation-Frames", f"pressure-Plt-Files"))

            # Create the animation file
            createVariableAnimation(temp_plt_dir,
                                    os.path.join(animation_dir, 'Pressure-Evolution-Animation.mp4'), fps=15)

        if check_flags['Domain State Animations'].get('Velocity', False):
            # Create directory for plt files
            temp_plt_dir = ensure_long_path_prefix(
                os.path.join(os.path.dirname(os.path.realpath(__file__)), f"Processed-Global-Results",
                             f"Animation-Frames", f"x_velocity-Plt-Files"))

            # Create the animation file
            createVariableAnimation(temp_plt_dir,
                                    os.path.join(animation_dir, 'X-Velocity-Evolution-Animation.mp4'), fps=15)

        if check_flags['Domain State Animations'].get('Species', False):
            for i in range(len(input_params.species)):
                temp_plt_dir = ensure_long_path_prefix(
                    os.path.join(os.path.dirname(os.path.realpath(__file__)), f"Processed-Global-Results",
                                 f"Animation-Frames", f"Y({str(input_params.species[i])})-Plt-Files"))

                # Create the animation file
                createVariableAnimation(temp_plt_dir,
                                        os.path.join(animation_dir,
                                                     f"Y({str(input_params.species[i])})-Evolution-Animation.mp4"),
                                        fps=15)

        if check_flags['Domain State Animations'].get('Heat Release Rate', False):
            # Create directory for plt files
            temp_plt_dir = ensure_long_path_prefix(
                os.path.join(os.path.dirname(os.path.realpath(__file__)), f"Processed-Global-Results",
                             f"Animation-Frames", f"HRR-Plt-Files"))

            # Create the animation file
            createVariableAnimation(temp_plt_dir,
                                    os.path.join(animation_dir, 'HRR-Evolution-Animation.mp4'), fps=15)

        if check_flags['Domain State Animations'].get('Combined', False):
            temp_plt_dir = ensure_long_path_prefix(
                os.path.join(os.path.dirname(os.path.realpath(__file__)), f"Processed-Global-Results",
                             f"Animation-Frames", f"{check_flags['Domain State Animations']['Combined'][0]}-"
                                                  f"{check_flags['Domain State Animations']['Combined'][1]}-Plt-Files"))

            # Create the animation file
            createVariableAnimation(temp_plt_dir, os.path.join(animation_dir,
                                                               f"{check_flags['Domain State Animations']['Combined'][0]}"
                                                               f"-{check_flags['Domain State Animations']['Combined'][1]}-Evolution-Animation.mp4"),
                                    fps=15)

        print('Completed Animation Processing')
    # Step 4: Write to file, if any of the sub-dictionary values except 'Domain State Animations' are true
    for key, sub_dict in check_flags.items():
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

        for master_key, sub_dict in check_flags.items():
            if 'Smoothing' in sub_dict and sub_dict['Smoothing'] == True:
                file_output(ensure_long_path_prefix(os.path.join(output_dir, 'Wave-Tracking-Smooth-Results.txt')), True)

    return


def pelec_ddt_processing_function(ddt_pltFile, plt_dir, domain_info, input_params, check_flags, output_dir=False):
    def parallel_processing_function(iter_arr, const_list, predicate, nProcs):
        """

        :param iter_arr:
        :param const_list:
        :param predicate:
        :param nProcs:
        :return:
        """
        # Perform the multiprocessing
        with multiprocessing.Pool(
                processes=nProcs, initargs=()
        ) as pool:
            y = pool.map(predicate,
                         zip(iter_arr,
                             itertools.repeat(const_list)))
        return y

    def file_output(file_path, spatial_cell, time_step, result_type=None):
        # Step 1: Dynamically create the text file header depending on the assigned flags
        if result_type == 'PeleC':
            header_data = ["Time [s]"]
            if check_flags['PeleC Processing'].get('Temperature', False):
                header_data.extend(["Temperature [K]"])
            if check_flags['PeleC Processing'].get('Pressure', False):
                header_data.extend(["Pressure [Pa]"])
            if check_flags['PeleC Processing'].get('Density', False):
                header_data.extend(["Density [kg/m3]"])
            if check_flags['PeleC Processing'].get('Species', False):
                for i in range(len(input_params.species)):
                    header_data.extend([f"Y({str(input_params.species[i])})"])
            if check_flags['PeleC Processing'].get('Heat Release Rate', False):
                header_data.extend(["Heat Release Rate [W/m3]"])

        elif result_type == 'Ignition Delay Pressure' or result_type == 'Ignition Delay Volume':
            header_data = ["Time [s]"]
            if check_flags['Ignition Delay Processing'].get('Temperature', False):
                header_data.extend(["Temperature [K]"])
            if check_flags['Ignition Delay Processing'].get('Pressure', False):
                header_data.extend(["Pressure [Pa]"])
            if check_flags['Ignition Delay Processing'].get('Density', False):
                header_data.extend(["Density [kg/m3]"])
            if check_flags['Ignition Delay Processing'].get('Species', False):
                for i in range(len(input_params.species)):
                    header_data.extend([f"Y({str(input_params.species[i])})"])
            if check_flags['Ignition Delay Processing'].get('Heat Release Rate', False):
                header_data.extend(["Heat Release Rate [W/m3]"])

        elif result_type == 'Flame':
            header_data = ["Position [m]"]
            if check_flags['Flame Processing'].get('Temperature', False):
                header_data.extend(["Temperature [K]"])
            if check_flags['Flame Processing'].get('Pressure', False):
                header_data.extend(["Pressure [Pa]"])
            if check_flags['Flame Processing'].get('Density', False):
                header_data.extend(["Density [kg/m3]"])
            if check_flags['Flame Processing'].get('Velocity', False):
                header_data.extend(["Velocity [m/s]"])
            if check_flags['Flame Processing'].get('Species', False):
                for i in range(len(input_params.species)):
                    header_data.extend([f"Y({str(input_params.species[i])})"])
            if check_flags['Flame Processing'].get('Heat Release Rate', False):
                header_data.extend(["Heat Release Rate [W/m3]"])

        #
        with (open(ensure_long_path_prefix(file_path), "w") as outfile):
            outfile.write("#")
            for i in range(len(header_data)):
                outfile.write("{0:<55.0f} ".format(int(i + 1)))
            outfile.write("\n#")
            for i in range(len(header_data)):
                outfile.write("{0:<55s} ".format(header_data[i]))
            outfile.write("\n")

            if result_type == 'PeleC':
                for i in range(len(plt_result)):
                    outfile.write(" {0:<55e}".format(plt_result[i]['Time']))
                    # Temperature
                    if check_flags['PeleC Processing'].get('Temperature', False):
                        outfile.write(" {0:<55e}".format(plt_result[i]['PeleC']['Temperature'][spatial_cell]))
                    # Pressure
                    if check_flags['PeleC Processing'].get('Pressure', False):
                        outfile.write(" {0:<55e}".format(plt_result[i]['PeleC']['Pressure'][spatial_cell]))
                    # Density
                    if check_flags['PeleC Processing'].get('Density', False):
                        outfile.write(" {0:<55e}".format(plt_result[i]['PeleC']['Density'][spatial_cell]))
                    # Species
                    if check_flags['PeleC Processing'].get('Species', False):
                        for j in range(len(input_params.species)):
                            outfile.write(" {0:<55e}".format(plt_result[i]['PeleC']['Species'][spatial_cell]))
                    # Heat Release Rate
                    if check_flags['PeleC Processing'].get('Heat Release Rate', False):
                        outfile.write(" {0:<55e}".format(plt_result[i]['PeleC']['Max Heat Release Rate'][spatial_cell]))

                    outfile.write("\n")

            elif result_type == 'Ignition Delay Pressure' or result_type == 'Ignition Delay Volume':
                if result_type == 'Ignition Delay Pressure':
                    for i in range(len(plt_result[time_step]['Ignition Delay Pressure'][spatial_cell]['Time'])):
                        outfile.write(" {0:<55e}".format(
                            plt_result[time_step]['Ignition Delay Pressure'][spatial_cell]['Time'][i]))
                        # Temperature
                        if check_flags['Ignition Delay Processing'].get('Temperature', False):
                            outfile.write(" {0:<55e}".format(
                                plt_result[time_step]['Ignition Delay Pressure'][spatial_cell]['Temperature'][i]))
                        # Pressure
                        if check_flags['Ignition Delay Processing'].get('Pressure', False):
                            outfile.write(" {0:<55e}".format(
                                plt_result[time_step]['Ignition Delay Pressure'][spatial_cell]['Pressure'][i]))
                        # Density
                        if check_flags['Ignition Delay Processing'].get('Density', False):
                            outfile.write(" {0:<55e}".format(
                                plt_result[time_step]['Ignition Delay Pressure'][spatial_cell]['Density'][i]))
                        # Species
                        if check_flags['Ignition Delay Processing'].get('Species', False):
                            for j in range(len(input_params.species)):
                                outfile.write(" {0:<55e}".format(
                                    plt_result[time_step]['Ignition Delay Pressure'][spatial_cell]['Species'][j]))
                        # Heat Release Rate
                        if check_flags['Ignition Delay Processing'].get('Heat Release Rate', False):
                            outfile.write(" {0:<55e}".format(
                                plt_result[time_step]['Ignition Delay Pressure'][spatial_cell]['Max Heat Release Rate'][
                                    i]))

                        outfile.write("\n")
                if result_type == 'Ignition Delay Volume':
                    for i in range(len(plt_result[time_step]['Ignition Delay Volume'][spatial_cell]['Time'])):
                        outfile.write(
                            " {0:<55e}".format(plt_result[time_step]['Ignition Delay Volume'][spatial_cell]['Time'][i]))
                        # Temperature
                        if check_flags['Ignition Delay Processing'].get('Temperature', False):
                            outfile.write(" {0:<55e}".format(
                                plt_result[time_step]['Ignition Delay Volume'][spatial_cell]['Temperature'][i]))
                        # Pressure
                        if check_flags['Ignition Delay Processing'].get('Pressure', False):
                            outfile.write(" {0:<55e}".format(
                                plt_result[time_step]['Ignition Delay Volume'][spatial_cell]['Pressure'][i]))
                        # Density
                        if check_flags['Ignition Delay Processing'].get('Density', False):
                            outfile.write(" {0:<55e}".format(
                                plt_result[time_step]['Ignition Delay Volume'][spatial_cell]['Density'][i]))
                        # Species
                        if check_flags['Ignition Delay Processing'].get('Species', False):
                            for j in range(len(input_params.species)):
                                outfile.write(" {0:<55e}".format(
                                    plt_result[time_step]['Ignition Delay Volume'][spatial_cell]['Species'][j]))
                        # Heat Release Rate
                        if check_flags['Ignition Delay Processing'].get('Heat Release Rate', False):
                            outfile.write(" {0:<55e}".format(
                                plt_result[time_step]['Ignition Delay Volume'][spatial_cell]['Max Heat Release Rate'][
                                    i]))

                        outfile.write("\n")
            elif result_type == 'Flame':
                for i in range(len(plt_result[time_step]['Flame'][spatial_cell]['Grid'])):
                    outfile.write(" {0:<55e}".format(plt_result[time_step]['Flame'][spatial_cell]['Grid'][i]))
                    # Temperature
                    if check_flags['Ignition Delay Processing'].get('Temperature', False):
                        outfile.write(
                            " {0:<55e}".format(plt_result[time_step]['Flame'][spatial_cell]['Temperature'][i]))
                    # Pressure
                    if check_flags['Ignition Delay Processing'].get('Pressure', False):
                        outfile.write(" {0:<55e}".format(plt_result[time_step]['Flame'][spatial_cell]['Pressure']))
                    # Density
                    if check_flags['Ignition Delay Processing'].get('Density', False):
                        outfile.write(" {0:<55e}".format(plt_result[time_step]['Flame'][spatial_cell]['Density'][i]))
                    # Velocity
                    if check_flags['Ignition Delay Processing'].get('Velocity', False):
                        outfile.write(" {0:<55e}".format(plt_result[time_step]['Flame'][spatial_cell]['Velocity'][i]))
                    # Species
                    if check_flags['Ignition Delay Processing'].get('Species', False):
                        for j in range(len(input_params.species)):
                            outfile.write(
                                " {0:<55e}".format(plt_result[time_step]['Flame'][spatial_cell]['Species'][j]))
                    # Heat Release Rate
                    if check_flags['Ignition Delay Processing'].get('Heat Release Rate', False):
                        outfile.write(" {0:<55e}".format(
                            plt_result[time_step]['Flame'][spatial_cell]['Max Heat Release Rate'][i]))

                    outfile.write("\n")
            outfile.close()
            return

    # Step 1: Determine the location of DDT
    temp_plt_data = yt.load(ddt_pltFile)
    max_level = temp_plt_data.index.max_level

    # Create sudo-grid at the maximum level present
    temp_ddt_data = temp_plt_data.covering_grid(level=max_level,
                                                left_edge=[0.0, 0.0, 0.0],
                                                dims=temp_plt_data.domain_dimensions * [2 ** max_level, 2 ** max_level,
                                                                                        1],
                                                # And any fields to preload (this is optional!)
                                                # fields=desired_varables
                                                )

    # Find max pressure location
    ddt_idx = np.unravel_index(np.argmax(temp_ddt_data["boxlib", 'Temp'].to_value(), axis=None),
                               temp_ddt_data["boxlib", 'Temp'].to_value().shape)

    ddt_window = np.empty(2, dtype=object)
    """
    ddt_window[0] = np.array([temp_ddt_data["boxlib", 'x'][ray_sort][ddt_idx[0] - check_flags['Processing Parameters']['Spatial Window'] // 2][0][0],
                              temp_ddt_data["boxlib", 'y'][ray_sort][0][ddt_idx[1]][0], 0.0])
    ddt_window[1] = np.array([temp_ddt_data["boxlib", 'x'][ray_sort][ddt_idx[0] + check_flags['Processing Parameters']['Spatial Window'] // 2][0][0],
                              temp_ddt_data["boxlib", 'y'][ray_sort][0][ddt_idx[1]][0], 0.0])
    """
    ddt_window[0] = [ddt_idx[0] - check_flags['Processing Parameters']['Spatial Window'] // 2, ddt_idx[1], 0]
    ddt_window[1] = [ddt_idx[0] + check_flags['Processing Parameters']['Spatial Window'] // 2, ddt_idx[1], 0]

    del temp_plt_data, temp_ddt_data
    # Step 1:
    print('Starting Raw Data Loading and Processing')
    plt_time_dir = plt_dir[
                   plt_dir.index(ddt_pltFile) - check_flags['Processing Parameters']['Time Window'][0]:plt_dir.index(
                       ddt_pltFile) + check_flags['Processing Parameters']['Time Window'][1]]
    plt_result = parallel_processing_function(plt_time_dir,
                                              (ddt_window, check_flags, [plt_dir[0], plt_dir[-1]], input_params,),
                                              single_ddt_pltfile_processing, n_proc)
    print('Completed Raw Data Loading and Processing')

    # Step 2:
    if os.path.exists(os.path.join(output_dir, f"Processed-DDT-Results")) is False:
        os.mkdir(os.path.join(output_dir, f"Processed-DDT-Results"))

    for spatial_cell in range(check_flags['Processing Parameters']['Spatial Window']):
        if os.path.exists(
                os.path.join(os.path.join(output_dir, f"Processed-DDT-Results", f"Cell-{spatial_cell}"))) is False:
            os.mkdir(
                ensure_long_path_prefix(os.path.join(output_dir, f"Processed-DDT-Results", f"Cell-{spatial_cell}")))

        if check_flags['Comparison Type'].get('PeleC', False):
            file_output(
                os.path.join(output_dir, f"Processed-DDT-Results", f"Cell-{spatial_cell}", f"PeleC-Results.dat"),
                spatial_cell, None, result_type='PeleC')

        for time_step in range(0, np.sum(check_flags['Processing Parameters']['Time Window'])):
            if check_flags['Comparison Type'].get('Ignition Delay Pressure', False):
                file_output(os.path.join(output_dir, f"Processed-DDT-Results", f"Cell-{spatial_cell}",
                                         f"Ignition-Delay-Const-Pressure-Time-Step-{time_step}.dat"),
                            spatial_cell, time_step, result_type='Ignition Delay Pressure')

            if check_flags['Comparison Type'].get('Ignition Delay Volume', False):
                file_output(os.path.join(output_dir, f"Processed-DDT-Results", f"Cell-{spatial_cell}",
                                         f"Ignition-Delay-Const-Volume-Time-Step-{time_step}.dat"),
                            spatial_cell, time_step, result_type='Ignition Delay Volume')

            if check_flags['Comparison Type'].get('Flame', False):
                file_output(os.path.join(output_dir, f"Processed-DDT-Results", f"Cell-{spatial_cell}",
                                         f"Flame-Time-Step-{time_step}.dat"),
                            spatial_cell, time_step, result_type='Flame')

    return


def main():
    ####################################################################################################################
    # This code is developed to process a 2D Planar Flame simulated using PeleC for a given y-position and a given
    # (temperature) isotherm for the flame and pressure for any shock
    #
    # All functions are configured for a 2 dimensional space
    ####################################################################################################################
    # Step 0: Set all the desired tasks to be performed bny the python script
    skip_load = 0
    row_index = "Middle"  # Desired row location for data collection
    ddt_plt_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Raw-PeleC-Data', 'plt426200')

    check_flag_dict = {
        'Flame Processing': {
            'Position': True,
            'Velocity': True,
            'Relative Velocity': False,
            'Thermodynamic State': True,
            'Heat Release Rate': True,
            'Flame Thickness': True,
            'Surface Length': True,
            'Smoothing': True
        },
        'Leading Shock Processing': {
            'Position': True,
            'Velocity': True,
            'Smoothing': True
        },
        'Maximum Pressure Processing': {
            'Position': False,
            'Thermodynamic State': False,
            'Smoothing': False
        },
        'Pre-Shock Processing': {
            'Thermodynamic State': True,
            'Smoothing': True
        },
        'Post-Shock Processing': {
            'Thermodynamic State': True,
            'Smoothing': True
        },
        'Domain State Animations': {
            'Temperature': True,
            'Pressure': True,
            'Velocity': True,
            'Species': True,
            'Heat Release Rate': True,
            'Flame Spectrum': False,
            'Combined': ('Temp', 'pressure')
        }
    }

    detonation_check_flag = {
        'Processing Parameters': {
            'Time Window': (10, 10),
            'Spatial Window': 5,
        },
        'Comparison Type': {
            'PeleC': True,
            'Ignition Delay Pressure': True,
            'Ignition Delay Volume': True,
            'Flame': True
        },
        'PeleC Processing': {
            'Temperature': True,
            'Pressure': True,
            'Density': True,
            'Species': False,
            'Heat Release Rate': False
        },
        'Ignition Delay Processing': {
            'Temperature': True,
            'Pressure': True,
            'Density': True,
            'Species': False,
            'Heat Release Rate': True
        },
        'Flame Processing': {
            'Temperature': True,
            'Pressure': True,
            'Density': True,
            'Velocity': True,
            'Species': True,
            'Heat Release Rate': True,
        }
    }

    # Step 1: Initialize the code with the desired processed variables and mixture composition
    input_params = MyClass()
    input_params.T = 503.15
    input_params.P = 10.0 * 100000
    input_params.Phi = 1.0
    input_params.Fuel = 'H2'
    input_params.mech = 'Li-Dryer-H2-mechanism.yaml'
    input_params.species = mechanism_species(input_params.mech)

    if input_params.Fuel == "H2":
        input_params.oxygenAmount = 0.5
    if input_params.Fuel == "C2H6":
        input_params.oxygenAmount = 3.5
    if input_params.Fuel == "C4H10":
        input_params.oxygenAmount = 6.5
    input_params.nitrogenAmount = 0
    input_params.X = {input_params.Fuel: input_params.Phi, 'O2': input_params.oxygenAmount,
                      'N2': input_params.nitrogenAmount}

    # Step 2: Create the result directories
    dir_path = os.path.dirname(os.path.realpath(__file__))
    if os.path.exists(os.path.join(dir_path, f"Processed-Global-Results")) is False:
        os.mkdir(os.path.join(dir_path, f"Processed-Global-Results"))
    output_dir_path = os.path.join(dir_path, f"Processed-Global-Results")

    # Step 3: Collect all the present pelec data directories
    time_data_dir = []
    for raw_data_folder in os.listdir(dir_path):
        if os.path.isdir(os.path.join(dir_path, raw_data_folder)) and raw_data_folder.startswith('Raw-PeleC'):
            raw_data_dir = os.path.join(dir_path, raw_data_folder)
            for time_step in os.listdir(raw_data_dir):
                if os.path.isdir(os.path.join(raw_data_dir, time_step)) and time_step.startswith('plt'):
                    time_data_dir.append(os.path.join(raw_data_dir, time_step))

    # Step 4: Chronologically order the pltFiles and truncate the raw data list if skip loading is enabled
    updated_data_list = sort_files(time_data_dir)

    if skip_load > 0:
        updated_data_list = updated_data_list[0::skip_load]

    # Step 5: Determine the domain sizing parameters (size, # of cells)
    domain_info = domain_size_parameters(updated_data_list[0], row_index)

    # Step 6: Individual PltFile Processing
    pelec_processing_function(updated_data_list, domain_info, input_params, check_flag_dict, output_dir=output_dir_path)

    # Step 7:
    # pelec_ddt_processing_function(ddt_plt_dir, updated_data_list, domain_info, input_params, detonation_check_flag, output_dir=output_dir_path)

    return


if __name__ == '__main__':
    main()