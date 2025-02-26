import os, re, yt, cv2, itertools, multiprocessing
import numpy as np
import cantera as ct
from ast import literal_eval
from itertools import groupby
import matplotlib.pyplot as plt
from sdtoolbox.thermo import soundspeed_fr
from sdtoolbox.postshock import PostShock_fr
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import BallTree


class MyClass():
    pass


def reload_prior_processed_data(raw_file_list, restart_file, check=False):
    def importGlobalData(fileName):
        # Step 0: Load the wave tracking data for time (0) to restart location
        data = np.loadtxt(fileName, encoding="utf8", unpack=True)
        return len(data[0, :])

    ####################################################################################################################
    # This function serves to determine what raw PeleC time step data files can be neglected due to prior processing
    # attempts
    ####################################################################################################################
    if check:
        pelec_time_step_len = len(raw_file_list) - 1
        restart_time_step_len = importGlobalData(restart_file) - 1
        updated_file_list = raw_file_list[restart_time_step_len:pelec_time_step_len]
        return updated_file_list
    else:
        return raw_file_list


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


def polynomial_fit_over_array(x, y, bin_size=51, degree=2):
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
    for i, point in enumerate(x):
        if i >= (bin_size + 1) / 2 and i <= (len(x) - (bin_size + 1) / 2):
            # Find the indices within the bin
            center_index = np.argmin(np.abs(x - point))
            start_index = max(0, center_index - bin_size // 2)
            end_index = min(len(x), center_index + (bin_size + 1) // 2)

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


def waveTrackingFunction(raw_data, ray_sort, tracking_str, tracking_var, wave_type):
    # Step 1: Load the desired marker and x positions
    temp_data = raw_data["boxlib", str(tracking_str)][ray_sort].to_value()
    temp_x_pos = raw_data["boxlib", str('x')][ray_sort].to_value()
    # Step 2:
    if wave_type == "Flame Processing":
        wave_index = np.argwhere(temp_data >= tracking_var)[-1][0]
    elif wave_type == "Leading Shock Processing" or wave_type == "Pre-Shock Processing" or wave_type == "Post-Shock Processing":
        # Compute the gradient of pressure with respect to space
        dp_dx = np.gradient(temp_data, temp_x_pos)
        # Find the indices where the gradient changes sign (indicating a shock)
        shock_indices = np.where(dp_dx == np.min(dp_dx))[0]
        # Classify shocks as left or right running
        wave_index = np.max(shock_indices)
    elif wave_type == "Maximum Pressure Processing":
        temp_index = temp_data
        wave_index = np.argwhere(temp_index == np.max(temp_index))[-1][0]
    return wave_index, temp_x_pos[wave_index] / 100


def wavePositionProcessingFunction(args):
    #####
    #
    #####
    # Step 1: Unpack the parallelization arguments
    current_file = args[0]
    domain_sizing = args[1][0]
    tracking_obj = args[1][1]
    # Step 2:
    domain_size = domain_sizing[0][1]
    # Step 3:
    raw_data = yt.load(current_file)
    slice = raw_data.ray(np.array([0.0, domain_size[1], 0.0]), np.array([domain_size[0], domain_size[1], 0.0]))
    ray_sort = np.argsort(slice["x"])
    time = raw_data.current_time.to_value()
    # Step 4:
    if len(tracking_obj) > 1:
        result_arr = np.zeros(len(tracking_obj) + 1)
        result_arr[0] = time
        for i, wave_type in enumerate(tracking_obj):
            if wave_type == "Flame Processing":
                tracking_str = "Temp"
                tracking_val = 2000
            else:
                tracking_str = "pressure"
                tracking_val = None
            # Now determine the wave position at the given time
            [_, temp_pos] = waveTrackingFunction(slice, ray_sort, tracking_str, tracking_val, wave_type)
            result_arr[i + 1] = temp_pos
    else:
        result_arr = np.zeros(2)
        result_arr[0] = time
        if tracking_obj[0] == "Flame Processing":
            tracking_str = "Temp"
            tracking_val = 2000
        else:
            tracking_str = "pressure"
            tracking_val = None
        [_, result_arr[1]] = waveTrackingFunction(slice, ray_sort, tracking_str, tracking_val, tracking_obj[0])

    return result_arr


def gasRelativeVelocityProcessingFunction(args):
    def preheatZoneFunction():
        # Step 1: Determine the reaction zone through temperature values
        [flame_index, _] = waveTrackingFunction(slice, ray_sort, "Temp", 2000, "Flame")
        # Step 2: Determine the leading shock location through the pressure gradient
        [shock_index, _] = waveTrackingFunction(slice, ray_sort, "pressure", None, "Shock")
        # Step 3: Using temperature data, truncate the data and determine where the temperature gradients go to near zero
        temp_data = slice["boxlib", str("Temp")][ray_sort].to_value()
        temp_index = np.argwhere(temp_data <= temp_data[flame_index] - 1000)[0][0]
        # Step 4:
        vel_data = slice["boxlib", str("x_velocity")][ray_sort].to_value() / 100
        return vel_data[temp_index]

    #####
    #
    #####
    # Step 0: Unpack the parallelization arguments
    current_file = args[0]
    domain_sizing = args[1][0]

    domain_size = domain_sizing[0][1]
    # Step 1:
    raw_data = yt.load(current_file)
    slice = raw_data.ray(np.array([0.0, domain_size[1], 0.0]), np.array([domain_size[0], domain_size[1],
                                                                         0.0]))  # 1 corresponds to the y-axis, y_location is the physical position (in meters)
    ray_sort = np.argsort(slice["x"])
    time = raw_data.current_time.to_value()
    # Step 2:
    gas_vel = preheatZoneFunction()
    return gas_vel


def thermodynamicStateProcessingFunction(args):
    def canteraGasObject(data, ray_sort, tracking_str, tracking_val, wave_type):
        # Step 1:
        [wave_index, _] = waveTrackingFunction(data, ray_sort, tracking_str, tracking_val, wave_type)
        temp_pos = data["boxlib", str('x')][ray_sort].to_value()[wave_index]
        if wave_type == "Flame Processing":
            probe_index = np.argwhere(data["boxlib", str('x')][ray_sort].to_value() >= temp_pos + (0.25 / 10))[0][0]
        elif wave_type == "Pre-Shock Processing" or wave_type == "Post-Shock Processing":
            if wave_type == "Pre-Shock Processing":
                try:
                    probe_index = np.argwhere(data["boxlib", str('x')][ray_sort].to_value() >= temp_pos + (5 / 10))[0][
                        0]
                except:
                    probe_index = len(data["boxlib", str('x')][ray_sort]) - 1
            elif wave_type == "Post-Shock Processing":
                try:
                    probe_index = np.argwhere(data["boxlib", str('x')][ray_sort].to_value() >= temp_pos - (5 / 10))[0][
                        0]
                except:
                    probe_index = len(data["boxlib", str('x')][ray_sort]) - 1
        elif wave_type == "Maximum Pressure Processing":
            try:
                probe_index = np.argwhere(data["boxlib", str('x')][ray_sort].to_value() >= temp_pos)[0][0]
            except:
                probe_index = len(data["boxlib", str('x')][ray_sort]) - 1

        # Step 2:
        temp_temp = data["boxlib", "Temp"][ray_sort].to_value()
        temp_pres = data["boxlib", "pressure"][ray_sort].to_value()

        temp_comp = []
        for i in range(len(input_params.result_species)):
            temp_comp.append(data["boxlib", str("Y(" + input_params.result_species[i] + ")")][ray_sort].to_value())

        mixture_comp = {}
        for i in range(len(input_params.result_species)):
            mixture_comp.update({str(input_params.result_species[i]): temp_comp[i][probe_index]})
        # Step 3:
        result_array = np.zeros(4, dtype=float)
        gas_obj = ct.Solution(input_params.mech)
        gas_obj.TPY = (temp_temp[probe_index],
                       temp_pres[probe_index] / 10,
                       mixture_comp)
        result_array[0] = gas_obj.T
        result_array[1] = gas_obj.P
        result_array[2] = gas_obj.density_mass
        result_array[3] = soundspeed_fr(gas_obj)
        del gas_obj
        return result_array.tolist()

    # Step 1:
    current_file = args[0]
    domain_sizing = args[1][0]
    input_params = args[1][1]
    tracking_obj = args[1][2]
    # Step 2:
    domain_size = domain_sizing[0][1]
    # Step 3:
    raw_data = yt.load(current_file)
    slice = raw_data.ray(np.array([0.0, domain_size[1], 0.0]), np.array([domain_size[0], domain_size[1], 0.0]))
    ray_sort = np.argsort(slice["x"])
    #
    if len(tracking_obj) > 1:
        result_arr = []
        for i, wave_type in enumerate(tracking_obj):
            if wave_type == "Flame Processing":
                tracking_str = "Temp"
                tracking_val = 2000
            else:
                tracking_str = "pressure"
                tracking_val = None
            # Now determine the wave position at the given time
            temp_arr = canteraGasObject(slice, ray_sort, tracking_str, tracking_val, wave_type)
            result_arr.append(temp_arr[:])
    else:
        if tracking_obj == "Flame Processing":
            tracking_str = "Temp"
            tracking_val = 2000
        else:
            tracking_str = "pressure"
            tracking_val = None
        result_arr = canteraGasObject(slice, ray_sort, tracking_str, tracking_val, tracking_obj)
    return result_arr


def flameAreaContourFunction(args):
    def sort_by_nearest_neighbors(points, plt_check=False):
        # Convert points to numpy array if it's a list of lists
        points = np.array(points)

        # Create NearestNeighbors instance
        nbrs = NearestNeighbors(n_neighbors=len(points), algorithm='ball_tree').fit(points)
        distances, indices = nbrs.kneighbors(points)

        # Construct the sorted order
        distance_metric = lambda x: (x[0] - 0) ** 2 + (x[1] - 0) ** 2
        origin_point = min(points, key=distance_metric)
        origin_idx = np.argwhere(points == origin_point)[0][0]

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
        print('Approximate Surface Length =', surface_approx, 'cm')
        # Check point order
        if plt_check:
            plt.figure(figsize=(8, 6))
            plt.plot(range(len(points)), order, marker='.', linestyle='-')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title('2D Isocontours')
            plt.show()

        return points[order]

    """

    :return: flame length for 2D simulation data
    """
    # Step 1: Unpack the arguments passed in from parallelization
    current_file = args[0]
    # Step 2: Load the current plt file
    raw_data = yt.load(current_file)
    raw_data.force_periodicity()
    all_data = raw_data.all_data()
    verts = all_data.extract_isocontours("Temp", 2000)
    rough_index = np.lexsort((verts[:, 0], verts[:, 1]))
    rough_sort = verts[rough_index]
    # Step 3: Remove outliers, artifacts of the periodicity, and order the array
    # Let us determine the buffer region caused by the periodic boundaries required
    buffer = 0.0125 * raw_data.domain_right_edge[1]
    y_lower = raw_data.domain_left_edge[1] + buffer
    y_upper = raw_data.domain_right_edge[1] - buffer
    # Handling 2D contours from extracted vertices
    # Assuming verts contains 3D coordinates (x, y, z), we'll extract the 2D data
    # verts is a list of arrays, each array contains (x, y, z) vertices
    contour_pts = np.empty((len(rough_sort), 2), dtype=object)
    for i, vert in enumerate(rough_sort):
        # Extract only x and y for 2D visualization for valid points
        if not np.isclose(vert[0], 0, atol=1e-02) and (y_lower <= vert[1] <= y_upper):
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
    contour_line = np.zeros(len(contour_arr) - 1)
    for i in range(1, len(contour_arr)):
        contour_line[i - 1] = np.sqrt((contour_arr[i, 0] - contour_arr[i - 1, 0]) ** 2 +
                                      (contour_arr[i, 1] - contour_arr[i - 1, 1]) ** 2)
    # Calculate the total line length, by summing the lines between points
    surface_length = np.sum(contour_line)
    print('Flame Surface Length: ', surface_length, 'cm')
    # Plot the 2D contours
    plt.figure(figsize=(8, 6))
    plt.plot(contour_arr[:, 0], contour_arr[:, 1], marker='.', linestyle='-')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('2D Iso-contours, Surface Length = {0:<10f} cm'.format(surface_length))
    plt.show()

    return surface_length


def pelecProcessingFunction(data_dir, domain_info, input_params, check_flags, output_dir=None):
    def parallelProcessingFunction(iter_arr, const_list, predicate, nProcs):
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

    def writeTextFile(file_path, smoothing_check):
        """

        :param file_path:
        :return:
        """
        # Step 1: Dynamically create the text file header depending on the assigned flags
        header_pos_flag = False
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
        # Flame Surface Length
        if check_flags['Flame Processing'].get('Surface Length', False):
            header_data.extend(["Flame Surface Length [m]"])

        # Step 2:
        with open(file_path, "w") as outfile:
            outfile.write("#")
            for i in range(len(header_data)):
                outfile.write("{0:<55s} ".format(header_data[i]))
            outfile.write("\n")
            # Step 2:
            # Step 2.1:
            if not smoothing_check:
                for i in range(len(time_arr)):
                    # Time
                    outfile.write(" {0:<55e}".format(time_arr[i]))

                    # Position
                    if check_flags['Flame Processing'].get('Position', False) and not compound_position_processing:
                        outfile.write(" {0:<55e}".format(flame_pos_arr[i]))

                    elif check_flags['Flame Processing'].get('Position', False) and compound_position_processing:
                        outfile.write(" {0:<55e}".format(comp_pos_arr[i, 0]))

                    if check_flags['Maximum Pressure Processing'].get('Position',
                                                                      False) and not compound_position_processing:
                        outfile.write(" {0:<55e}".format(max_pres_pos_arr[i]))

                    elif check_flags['Maximum Pressure Processing'].get('Position',
                                                                        False) and compound_position_processing:
                        outfile.write(" {0:<55e}".format(comp_pos_arr[i, 1]))

                    if check_flags['Leading Shock Processing'].get('Position',
                                                                   False) and not compound_position_processing:
                        outfile.write(" {0:<55e}".format(lead_shock_pos_arr[i, 1]))

                    elif check_flags['Leading Shock Processing'].get('Position',
                                                                     False) and compound_position_processing:
                        outfile.write(" {0:<55e}".format(comp_pos_arr[i, 2]))

                    # Velocity
                    if check_flags['Flame Processing'].get('Velocity', False):
                        outfile.write(" {0:<55e}".format(flame_vel_arr[i]))

                    if check_flags['Maximum Pressure Processing'].get('Velocity', False):
                        outfile.write(" {0:<55e}".format(lead_shock_vel_arr[i]))

                    # Relative Velocity
                    if check_flags['Flame Processing'].get('Relative Velocity', False):
                        outfile.write(" {0:<55e}".format(flame_rel_vel_arr[i]))

                    # Thermodynamic State
                    if check_flags['Flame Processing'].get('Thermodynamic State',
                                                           False) and not compound_thermodynamic_processing:
                        for j in range(len(flame_thermo_arr[0, :])):
                            outfile.write(" {0:<55e}".format(flame_thermo_arr[i, j]))

                    elif check_flags['Flame Processing'].get('Thermodynamic State',
                                                             False) and compound_thermodynamic_processing:
                        for j in range(len(comp_thermo_arr[0][0])):
                            outfile.write(" {0:<55e}".format(comp_thermo_arr[
                                                                 i, get_index_of_string(sub_dicts_with_position,
                                                                                        'Flame Processing'), j]))

                    if check_flags['Maximum Pressure Processing'].get('Thermodynamic State',
                                                                      False) and not compound_thermodynamic_processing:
                        for j in range(len(max_pres_thermo_arr[0, :])):
                            outfile.write(" {0:<55e}".format(max_pres_thermo_arr[i, j]))

                    elif check_flags['Maximum Pressure Processing'].get('Thermodynamic State',
                                                                        False) and compound_thermodynamic_processing:
                        for j in range(len(comp_thermo_arr[0][0])):
                            outfile.write(" {0:<55e}".format(comp_thermo_arr[
                                                                 i, get_index_of_string(sub_dicts_with_position,
                                                                                        'Maximum Pressure Processing'), j]))

                    if check_flags['Pre-Shock Processing'].get('Thermodynamic State',
                                                               False) and not compound_thermodynamic_processing:
                        for j in range(len(pre_shock_thermo_arr[0, :])):
                            outfile.write(" {0:<55e}".format(pre_shock_thermo_arr[i, j]))

                    elif check_flags['Pre-Shock Processing'].get('Thermodynamic State',
                                                                 False) and compound_thermodynamic_processing:
                        for j in range(len(comp_thermo_arr[0][0])):
                            outfile.write(" {0:<55e}".format(comp_thermo_arr[
                                                                 i, get_index_of_string(sub_dicts_with_position,
                                                                                        'Pre-Shock Processing'), j]))

                    if check_flags['Post-Shock Processing'].get('Thermodynamic State',
                                                                False) and not compound_thermodynamic_processing:
                        for j in range(len(post_shock_thermo_arr[0, :])):
                            outfile.write(" {0:<55e}".format(post_shock_thermo_arr[i, j]))

                    elif check_flags['Post-Shock Processing'].get('Thermodynamic State',
                                                                  False) and compound_thermodynamic_processing:
                        for j in range(len(comp_thermo_arr[0][0])):
                            outfile.write(" {0:<55e}".format(comp_thermo_arr[
                                                                 i, get_index_of_string(sub_dicts_with_position,
                                                                                        'Post-Shock Processing'), j]))

                    # Surface Length
                    if check_flags['Flame Processing'].get('Surface Length', False):
                        outfile.write(" {0:<55e}".format(flame_surf_len_arr[i]))
                    outfile.write("\n")
            # Step 2.2:
            if smoothing_check:
                for i in range(len(smooth_time_arr)):
                    # Time
                    outfile.write(" {0:<55e}".format(smooth_time_arr[i]))

                    # Position
                    if check_flags['Flame Processing'].get('Position', False):
                        outfile.write(" {0:<55e}".format(smooth_flame_pos_arr[i, 1]))

                    if check_flags['Maximum Pressure Processing'].get('Position', False):
                        outfile.write(" {0:<55e}".format(smooth_lead_shock_pos_arr[i, 1]))

                    if check_flags['Pre-Shock Processing'].get('Position', False):
                        outfile.write(" {0:<55e}".format(smooth_max_pres_pos_arr[i, 1]))

                    # Velocity
                    if check_flags['Flame Processing'].get('Velocity', False):
                        outfile.write(" {0:<55e}".format(smooth_flame_vel_arr[i]))

                    if check_flags['Maximum Pressure Processing'].get('Velocity', False):
                        outfile.write(" {0:<55e}".format(smooth_lead_shock_vel_arr[i]))

                    # Relative Velocity
                    if check_flags['Flame Processing'].get('Relative Velocity', False):
                        outfile.write(" {0:<55e}".format(smooth_flame_rel_vel_arr[i]))

                    # Thermodynamic State
                    if check_flags['Flame Processing'].get('Thermodynamic State', False):
                        for j in range(len(smooth_flame_thermo_arr[0, :])):
                            outfile.write(" {0:<55e}".format(smooth_flame_thermo_arr[i, j]))

                    if check_flags['Maximum Pressure Processing'].get('Thermodynamic State', False):
                        for j in range(len(smooth_max_pres_thermo_arr[0, :])):
                            outfile.write(" {0:<55e}".format(smooth_max_pres_thermo_arr[i, j]))

                    if check_flags['Pre-Shock Processing'].get('Thermodynamic State', False):
                        for j in range(len(smooth_pre_shock_thermo_arr[0, :])):
                            outfile.write(" {0:<55e}".format(smooth_pre_shock_thermo_arr[i, j]))

                    if check_flags['Post-Shock Processing'].get('Thermodynamic State', False):
                        for j in range(len(smooth_post_shock_thermo_arr[0, :])):
                            outfile.write(" {0:<55e}".format(smooth_post_shock_thermo_arr[i, j]))

                    # Surface Length
                    if check_flags['Flame Processing'].get('Surface Length', False):
                        outfile.write(" {0:<55e}".format(smooth_flame_surf_len_arr[i]))
                    outfile.write("\n")
            outfile.close()
        return

    def get_index_of_string(array, string):
        try:
            return array.index(string)
        except ValueError:
            return -1  # Or any other indication that the string was not found

    """

    """
    # Step 0:

    # Step 1:

    # Step 2: Compound processing to reduce the number of data calls
    # Step 2.1: Position
    compound_position_processing = False
    sub_dicts_with_position = [sub_dict_name for sub_dict_name, sub_dict in check_flags.items() if
                               sub_dict.get('Position', False)]
    if len(sub_dicts_with_position) > 1:
        compound_position_processing = True
        temp_arr = parallelProcessingFunction(data_dir, (domain_info, sub_dicts_with_position),
                                              wavePositionProcessingFunction, 12)
        comp_pos_arr = np.array(temp_arr)
        del temp_arr
    # Step 2.2: Thermodynamic State
    compound_thermodynamic_processing = False
    sub_dicts_with_thermo = [sub_dict_name for sub_dict_name, sub_dict in check_flags.items() if
                             sub_dict.get('Thermodynamic State', False)]
    if len(sub_dicts_with_thermo) > 1:
        compound_thermodynamic_processing = True
        temp_arr = parallelProcessingFunction(data_dir, (domain_info, input_params, sub_dicts_with_thermo),
                                              thermodynamicStateProcessingFunction, 12)
        comp_thermo_arr = np.array(temp_arr)
        del temp_arr

    # Step 3:
    # Step 3.1: Flame Processing
    if 'Flame Processing' in check_flags:
        # Position
        if check_flags['Flame Processing'].get('Position', False) and not compound_position_processing:
            temp_arr = parallelProcessingFunction(data_dir, (domain_info, ['Flame Processing'],),
                                                  wavePositionProcessingFunction, 12)
            flame_pos_arr = np.array(temp_arr)[:, 1]
            time_arr = temp_arr[:, 0]
            del temp_arr
        elif check_flags['Flame Processing'].get('Position', False) and compound_position_processing:
            flame_pos_arr = comp_pos_arr[:, get_index_of_string(sub_dicts_with_position, 'Flame Processing')]
            time_arr = comp_pos_arr[:, 0]

        # Velocity
        if check_flags['Flame Processing'].get('Velocity', False):
            if check_flags['Flame Processing'].get('Position', True):
                print('Error: Must determine position in order to calculate velocity')
            else:
                flame_vel_arr = np.gradient(flame_pos_arr[:, 1]) / np.gradient(flame_pos_arr[:, 0])

        # Relative Velocity
        if check_flags['Flame Processing'].get('Relative Velocity', False):
            temp_arr = parallelProcessingFunction(data_dir, (domain_info,),
                                                  gasRelativeVelocityProcessingFunction, 12)
            flame_rel_vel_arr = np.array(temp_arr)
            del temp_arr

        # Thermodynamic State
        if check_flags['Flame Processing'].get('Thermodynamic State', False) and not compound_thermodynamic_processing:
            temp_arr = parallelProcessingFunction(data_dir, (domain_info, input_params, ['Flame Processing'],),
                                                  thermodynamicStateProcessingFunction, 12)
            flame_thermo_arr = np.array(temp_arr)
            del temp_arr
        elif check_flags['Flame Processing'].get('Thermodynamic State', False) and compound_thermodynamic_processing:
            temp_arr = comp_thermo_arr[:, get_index_of_string(sub_dicts_with_thermo, 'Flame Processing')]
            flame_thermo_arr = np.array(temp_arr)
            del temp_arr

        # Surface Length
        if check_flags['Flame Processing'].get('Surface Length', False):
            temp_arr = parallelProcessingFunction(data_dir, (), flameAreaContourFunction, 12)
            flame_surf_len_arr = np.array(temp_arr)
            del temp_arr

        # Smoothing
        if check_flags['Flame Processing'].get('Smoothing', False):
            # Position
            if check_flags['Flame Processing'].get('Position', False):
                [smooth_time_arr, smooth_flame_pos_arr, _] = polynomial_fit_over_array(time_arr, flame_pos_arr)
            # Velocity
            if check_flags['Flame Processing'].get('Velocity', False):
                [_, smooth_flame_vel_arr, _] = polynomial_fit_over_array(time_arr, flame_vel_arr)
            # Relative Velocity
            if check_flags['Flame Processing'].get('Relative Velocity', False):
                [_, smooth_flame_rel_vel_arr, _] = polynomial_fit_over_array(time_arr, flame_rel_vel_arr)
            # Thermodynamic State
            if check_flags['Flame Processing'].get('Thermodynamic State', False):
                smooth_flame_thermo_arr = np.zeros((len(smooth_time_arr), 4))
                for i in range(0, 3):
                    [_, temp_arr, _] = polynomial_fit_over_array(time_arr, flame_thermo_arr[:, i])
                    smooth_flame_thermo_arr[:, i] = temp_arr
            # Surface Length
            if check_flags['Flame Processing'].get('Surface Length', False):
                [_, smooth_flame_surf_len_arr, _] = polynomial_fit_over_array(time_arr, flame_surf_len_arr)

    # Step 3.2: Leading Shock Wave Processing
    if 'Leading Shock Processing' in check_flags:
        # Position
        if check_flags['Leading Shock Processing'].get('Position', False) and not compound_position_processing:
            temp_arr = parallelProcessingFunction(data_dir, (domain_info, ['Leading Shock Processing'],),
                                                  wavePositionProcessingFunction, 12)
            lead_shock_pos_arr = np.array(temp_arr)[:, 1]
            if 'time_arr' not in locals():
                time_arr = temp_arr[:, 0]
            del temp_arr
        elif check_flags['Leading Shock Processing'].get('Position', False) and compound_position_processing:
            lead_shock_pos_arr = comp_pos_arr[:,
                                 get_index_of_string(sub_dicts_with_position, 'Leading Shock Processing')]
            if 'time_arr' not in locals():
                time_arr = comp_pos_arr[:, 0]

        # Velocity
        if check_flags['Leading Shock Processing'].get('Velocity', False):
            if check_flags['Leading Shock Processing'].get('Position', True):
                print('Error: Must determine position in order to calculate velocity')
            else:
                lead_shock_vel_arr = np.gradient(lead_shock_pos_arr[:, 0]) / np.gradient(time_arr)

        # Smoothing
        if check_flags['Leading Shock Processing'].get('Smoothing', False):
            # Position
            if check_flags['Leading Shock Processing'].get('Position', False):
                [temp_time_arr, smooth_lead_shock_pos_arr, _] = polynomial_fit_over_array(time_arr, lead_shock_pos_arr)
                if 'smooth_time_arr' not in locals():
                    smooth_time_arr = temp_time_arr
                del temp_time_arr

            # Velocity
            if check_flags['Leading Shock Processing'].get('Velocity', False):
                [_, smooth_lead_shock_vel_arr, _] = polynomial_fit_over_array(time_arr, lead_shock_vel_arr)
                if 'smooth_time_arr' not in locals():
                    smooth_time_arr = temp_time_arr
                del temp_time_arr

    # Step 3.2: Maximum Pressure Processing
    if 'Maximum Pressure Processing' in check_flags:
        # Position
        if check_flags['Maximum Pressure Processing'].get('Position', False) and not compound_position_processing:
            temp_arr = parallelProcessingFunction(data_dir, (domain_info, ['Maximum Pressure Processing'],),
                                                  wavePositionProcessingFunction, 12)
            max_pres_pos_arr = np.array(temp_arr)[:, 1]
            if 'time_arr' not in locals():
                time_arr = temp_arr[:, 0]
            del temp_arr
        elif check_flags['Maximum Pressure Processing'].get('Position', False) and compound_position_processing:
            max_pres_pos_arr = comp_pos_arr[:,
                               get_index_of_string(sub_dicts_with_position, 'Maximum Pressure Processing')]
            if 'time_arr' not in locals():
                time_arr = comp_pos_arr[:, 0]

        # Thermodynamic State
        if check_flags['Maximum Pressure Processing'].get('Thermodynamic State',
                                                          False) and not compound_thermodynamic_processing:
            temp_arr = parallelProcessingFunction(data_dir,
                                                  (domain_info, input_params, ['Maximum Pressure Processing'],),
                                                  thermodynamicStateProcessingFunction, 12)
            max_pres_thermo_arr = np.array(temp_arr)
            del temp_arr
        elif check_flags['Maximum Pressure Processing'].get('Thermodynamic State',
                                                            False) and compound_thermodynamic_processing:
            temp_arr = comp_thermo_arr[:, get_index_of_string(sub_dicts_with_thermo, 'Maximum Pressure Processing')]
            max_pres_thermo_arr = np.array(temp_arr)
            del temp_arr

        # Smoothing
        if check_flags['Maximum Pressure Processing'].get('Smoothing', False):
            # Position
            if check_flags['Maximum Pressure Processing'].get('Position', False):
                [temp_time_arr, smooth_max_pres_pos_arr, _] = polynomial_fit_over_array(time_arr, max_pres_pos_arr)
                if 'smooth_time_arr' not in locals():
                    smooth_time_arr = temp_time_arr
                del temp_time_arr
            # Thermodynamic State
            if check_flags['Maximum Pressure Processing'].get('Thermodynamic State', False):
                smooth_max_pres_thermo_arr = np.zeros((len(smooth_time_arr), 4))
                for i in range(0, 3):
                    [_, temp_arr, _] = polynomial_fit_over_array(time_arr, max_pres_thermo_arr)
                    smooth_max_pres_thermo_arr[:, i] = temp_arr

    # Step 3.3: Pre-Shock Processing
    if 'Pre-Shock Processing' in check_flags:
        # Thermodynamic State
        if check_flags['Pre-Shock Processing'].get('Thermodynamic State',
                                                   False) and not compound_thermodynamic_processing:
            temp_arr = parallelProcessingFunction(data_dir, (domain_info, input_params, ['Pre-Shock Processing'],),
                                                  thermodynamicStateProcessingFunction, 12)
            pre_shock_thermo_arr = np.array(temp_arr)
            del temp_arr
        elif check_flags['Pre-Shock Processing'].get('Thermodynamic State',
                                                     False) and compound_thermodynamic_processing:
            temp_arr = comp_thermo_arr[:, get_index_of_string(sub_dicts_with_thermo, 'Pre-Shock Processing')]
            pre_shock_thermo_arr = np.array(temp_arr)
            del temp_arr

        # Smoothing
        if check_flags['Pre-Shock Processing'].get('Smoothing', False):
            if check_flags['Pre-Shock Processing'].get('Thermodynamic State', False):
                # Thermodynamic State
                smooth_pre_shock_thermo_arr = np.zeros((len(smooth_time_arr), 4))
                for i in range(0, 3):
                    [_, temp_arr, _] = polynomial_fit_over_array(time_arr, pre_shock_thermo_arr)
                    smooth_pre_shock_thermo_arr[:, i] = temp_arr

    # Step 3.4: Post-Shock Processing
    if 'Post-Shock Processing' in check_flags:
        # Thermodynamic State
        if check_flags['Post-Shock Processing'].get('Thermodynamic State',
                                                    False) and not compound_thermodynamic_processing:
            temp_arr = parallelProcessingFunction(data_dir, (domain_info, input_params, ['Post-Shock Processing'],),
                                                  thermodynamicStateProcessingFunction, 12)
            post_shock_thermo_arr = np.array(temp_arr)
            del temp_arr
        elif check_flags['Post-Shock Processing'].get('Thermodynamic State',
                                                      False) and compound_thermodynamic_processing:
            temp_arr = comp_thermo_arr[:, get_index_of_string(sub_dicts_with_thermo, 'Post-Shock Processing')]
            post_shock_thermo_arr = np.array(temp_arr)
            del temp_arr

        # Smoothing
        if check_flags['Post-Shock Processing'].get('Smoothing', False):
            # Thermodynamic State
            if check_flags['Post-Shock Processing'].get('Thermodynamic State', False):
                smooth_post_shock_thermo_arr = np.zeros((len(smooth_time_arr), 4))
                for i in range(0, 3):
                    [_, temp_arr, _] = polynomial_fit_over_array(time_arr, post_shock_thermo_arr)
                    smooth_post_shock_thermo_arr[:, i] = temp_arr

    # Step 4:

    # Step 5:
    # Step 5.1:
    writeTextFile(os.path.join(output_dir, 'Wave-Tracking-Results.txt'), False)
    # Step 5.2:
    if check_flags['Flame Processing'].get('Smoothing', False):
        writeTextFile(os.path.join(output_dir, 'Wave-Tracking-Smooth-Results.txt'), True)

    return


def main():
    ####################################################################################################################
    # This code is developed to process a 2D Planar Flame simulated using PeleC for a given y-position and a given
    # (temperature) isotherm for the flame and pressure for any shock
    #
    # All functions are configured for a 2 dimensional space
    ####################################################################################################################
    # Step 0: Set all the desired tasks to be performed bny the python script
    reload_data = False

    check_flag_dict = {
        'Flame Processing': {
            'Position': True,
            'Velocity': False,
            'Relative Velocity': False,
            'Thermodynamic State': True,
            'Surface Length': True,
            'Smoothing': True
        },
        'Leading Shock Processing': {
            'Position': True,
            'Velocity': False,
        },
        'Maximum Pressure Processing': {
            'Position': False,
            'Thermodynamic State': False,
            'Smoothing': False
        },
        'Pre-Shock Processing': {
            'Thermodynamic State': True,
            'Smoothing': False
        },
        'Post-Shock Processing': {
            'Thermodynamic State': False,
            'Smoothing': False
        }
    }

    skip_load = 10  # 0 for no skip
    # row_index = "Middle"  # Desired row location for data collection
    row_index = 0.045  # Desired y_location for data collection in cm
    mass_fraction_variables = np.array(["H", "H2", "H2O", "H2O2", "HO2", "N2", "O", "O2", "OH"])

    # Step 1: Initialize the code with the desired processed variables and mixture composition
    input_params = MyClass()
    input_params.T = 503.15
    input_params.P = 10.0 * ct.one_atm
    input_params.Phi = 1.0
    input_params.Fuel = 'H2'
    input_params.result_species = mass_fraction_variables

    if input_params.Fuel == "H2":
        input_params.oxygenAmount = 0.5
    if input_params.Fuel == "C2H6":
        input_params.oxygenAmount = 3.5
    if input_params.Fuel == "C4H10":
        input_params.oxygenAmount = 6.5
    input_params.nitrogenAmount = 0
    input_params.X = {input_params.Fuel: input_params.Phi, 'O2': input_params.oxygenAmount,
                      'N2': input_params.nitrogenAmount}
    input_params.mech = 'Li-Dryer-H2-mechanism.yaml'
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
    # Step 4: Check to see if processing restart is enabled
    updated_data_list = reload_prior_processed_data(time_data_dir,
                                                    os.path.join(output_dir_path, f"Wave-Tracking-Results.txt"),
                                                    check=reload_data)
    # Step 5: Truncate the raw data list if skip loading is enabled
    if skip_load > 0:
        updated_data_list = updated_data_list[0::skip_load]
    # Step 5:
    domain_sizing = domain_size_parameters(updated_data_list[0], row_index)

    # Step 6:
    pelecProcessingFunction(updated_data_list, domain_sizing, input_params, check_flag_dict, output_dir=output_dir_path)

    return


if __name__ == "__main__":
    main()