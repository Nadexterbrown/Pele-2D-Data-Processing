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


def polynomial_fit_over_array(x, y, bin_size, degree):
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
    if wave_type == "Flame":
        wave_index = np.argwhere(temp_data >= tracking_var)[-1][0]
    elif wave_type == "Shock":
        # Compute the gradient of pressure with respect to space
        dp_dx = np.gradient(temp_data, temp_x_pos)
        # Find the indices where the gradient changes sign (indicating a shock)
        shock_indices = np.where(dp_dx == np.min(dp_dx))[0]
        # Classify shocks as left or right running
        wave_index = np.max(shock_indices)
    elif wave_type == "Max Pressure":
        temp_index = temp_data
        wave_index = np.argwhere(temp_index == np.max(temp_index))[-1][0]
    return wave_index, temp_x_pos[wave_index] / 100


def wavePositionProcessingFunction(args):
    #####
    #
    #####
    # Step 0: Unpack the parallelization arguments
    current_file = args[0]
    domain_sizing = args[1]

    domain_size = domain_sizing[0][1]
    # Step 1:
    raw_data = yt.load(current_file)
    slice = raw_data.ray(np.array([0.0, domain_size[1], 0.0]), np.array([domain_size[0], domain_size[1], 0.0]))
    ray_sort = np.argsort(slice["x"])
    time = raw_data.current_time.to_value()
    # Step 3:
    [_, flame_position] = waveTrackingFunction(slice, ray_sort, "Temp", 2000, "Flame")
    [_, shock_position] = waveTrackingFunction(slice, ray_sort, "pressure", None, "Shock")
    [_, max_pressure_position] = waveTrackingFunction(slice, ray_sort, "pressure", None, "Max Pressure")
    wave_array = np.array([time, flame_position, shock_position, max_pressure_position])

    return wave_array


def gasVelocityProcessingFunction(args):
    def preheatZoneFunction(data, ray_sort):
        # Step 1: Determine the reaction zone through temperature values
        [flame_index, _] = waveTrackingFunction(data, ray_sort, "Temp", 2000, "Flame")
        # Step 2: Determine the leading shock location through the pressure gradient
        [shock_index, _] = waveTrackingFunction(data, ray_sort, "pressure", None, "Shock")
        # Step 3: Using temperature data, truncate the data and determine where the temperature gradients go to near zero
        temp_data = data["boxlib", str("Temp")][ray_sort].to_value()
        temp_index = np.argwhere(temp_data <= temp_data[flame_index] - 1000)[0][0]
        # Step 4:
        vel_data = data["boxlib", str("x_velocity")][ray_sort].to_value() / 100
        return vel_data[temp_index]

    #####
    #
    #####
    # Step 0: Unpack the parallelization arguments
    current_file = args[0]
    domain_sizing = args[1]

    domain_size = domain_sizing[0][1]
    # Step 1:
    raw_data = yt.load(current_file)
    slice = raw_data.ray(np.array([0.0, domain_size[1], 0.0]), np.array([domain_size[0], domain_size[1],
                                                                         0.0]))  # 1 corresponds to the y-axis, y_location is the physical position (in meters)
    ray_sort = np.argsort(slice["x"])
    time = raw_data.current_time.to_value()
    # Step 2:
    gas_vel = preheatZoneFunction(slice, ray_sort)
    return gas_vel


def thermodynamicProcessingFunction(args):
    def thermodynamicParameters(data, ray_sort, input_params, thermo_type):
        # Step 0: Determine the location of the desired wave provided by the thermo_type
        if thermo_type == "flame":
            [wave_index, _] = waveTrackingFunction(data, ray_sort, "Temp", 2000, "Flame")
            temp_array = data["boxlib", str('x')][ray_sort].to_value()
            temp_position = temp_array[wave_index]
            probe_index = np.argwhere(temp_array >= temp_position + (0.25 / 10))[0][0]
        elif thermo_type == "pre_shock" or thermo_type == "post_shock":
            [wave_index, _] = waveTrackingFunction(data, ray_sort, "pressure", None, "Shock")
            temp_array = data["boxlib", str('x')][ray_sort].to_value()
            temp_position = temp_array[wave_index]
            if thermo_type == "pre_shock":
                probe_index = np.argwhere(temp_array >= temp_position + (5 / 10))
            elif thermo_type == "post_shock":
                probe_index = np.argwhere(temp_array >= temp_position - (5 / 10))
        # Step 1:
        temp_temp = data["boxlib", "Temp"][ray_sort].to_value()
        temp_pres = data["boxlib", "pressure"][ray_sort].to_value()

        temp_comp = []
        for i in range(len(input_params.result_species)):
            temp_comp.append(data["boxlib", str("Y(" + input_params.result_species[i] + ")")][ray_sort].to_value())

        mixture_comp = {}
        for i in range(len(input_params.result_species)):
            mixture_comp.update({str(input_params.result_species[i]): temp_comp[i][probe_index]})

        # Step 0: Allocate the space for all thermodynamic parameters (reactant temperature, pressure, density, soundspeed
        # and the coresponding product values)
        result_array = np.empty(4, dtype=float)
        # Step 1: Using the input parameter class, create a cantera gas object
        gas_temp = ct.Solution(input_params.mech)
        gas_temp.TPY = (temp_temp[probe_index],
                        temp_pres[probe_index] / 10,
                        mixture_comp)
        result_array[0] = gas_temp.T
        result_array[1] = gas_temp.P
        result_array[2] = gas_temp.density_mass
        result_array[3] = soundspeed_fr(gas_temp)
        del gas_temp
        return result_array

    #####
    #
    #####
    # Step 1:
    current_file = args[0]
    domain_sizing = args[1]
    input_params = args[2]
    thermo_type = args[3]

    domain_size = domain_sizing[0][1]
    # Step 2: Determine the file path for the highest level
    raw_data = yt.load(current_file)
    slice = raw_data.ray(np.array([0.0, domain_size[1], 0.0]), np.array([domain_size[0], domain_size[1],
                                                                         0.0]))  # 1 corresponds to the y-axis, y_location is the physical position (in meters)
    ray_sort = np.argsort(slice["x"])
    time = raw_data.current_time.to_value()
    # Step 4:
    thermo_results = thermodynamicParameters(slice, ray_sort, input_params, thermo_type)
    return thermo_results


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

    return


def waveProcessingFunction(data_dir, row_index, input_params, check_flag=None,
                           output_dir=None):
    def parallelWaveProcessingFunction(paralellizationList, y_location, predicate, nProcs):
        """
            Call the function ``predicate`` on ``nProcs`` processors for ``nTemps``
            different temperatures.
        """
        # Perform the multiprocessing
        with multiprocessing.Pool(
                processes=nProcs, initargs=()
        ) as pool:
            y = pool.map(predicate,
                         zip(paralellizationList,
                             itertools.repeat(y_location)))
        return y

    def parallelRelativeProcessingFunction(paralellizationList, y_location, predicate, nProcs):
        """
                    Call the function ``predicate`` on ``nProcs`` processors for ``nTemps``
                    different temperatures.
                """
        # Perform the multiprocessing
        with multiprocessing.Pool(
                processes=nProcs, initargs=()
        ) as pool:
            y = pool.map(predicate,
                         zip(paralellizationList,
                             itertools.repeat(y_location)))
        return y

    def parallelThermodynamicProcessingFunction(paralellizationList, y_location, input_parameters, thermo_type,
                                                predicate, nProcs):
        """
                    Call the function ``predicate`` on ``nProcs`` processors for ``nTemps``
                    different temperatures.
                """
        # Perform the multiprocessing
        with multiprocessing.Pool(
                processes=nProcs, initargs=()
        ) as pool:
            y = pool.map(predicate,
                         zip(paralellizationList,
                             itertools.repeat(y_location),
                             itertools.repeat(input_parameters),
                             itertools.repeat(thermo_type)))
        return y

    def parallelFlameAreaProcessingFunction(paralellizationList, y_location, predicate, nProcs):
        """
                    Call the function ``predicate`` on ``nProcs`` processors for ``nTemps``
                    different temperatures.
                """
        # Perform the multiprocessing
        with multiprocessing.Pool(
                processes=nProcs, initargs=()
        ) as pool:
            y = pool.map(predicate,
                         zip(paralellizationList,
                             itertools.repeat(y_location)))
        return y

    def writeWaveResults(file_path, smooth_check, check_flag, position_data, lab_ref_vel, rel_ref_vel,
                         flame_thermo_data, pre_shock_thermo_data,
                         post_shock_thermo_data):
        # Step 0:
        header_data = ["Time [s]"]
        if check_flag.get('Position Flag', False):
            header_data.extend(["Flame Position [m]", "Shock Position [m]", "Max Pressure Position [m]"])
        if check_flag.get('Lab Velocity Flag', False):
            header_data.extend(["Flame Lab Velocity [m]", "Shock Lab Velocity [m]"])
        if check_flag.get('Relative Velocity Flag', False):
            header_data.extend(["Gas Velocity [m/s]", "Flame Ref Velocity [m]", "Shock Ref Velocity [m]"])
        if check_flag.get('Flame Thermodynamic State', False):
            header_data.extend(["Flame Temperature [K]", "Flame Pressure [Pa]", "Flame Density [kg/m^3]",
                                "Flame Soundspeed [m/s]"])
        if check_flag.get('Pre-Shock Thermodynamic State', False):
            header_data.extend(["Pre-Shock Temperature [K]", "Pre-Shock Pressure [Pa]", "Pre-Shock Density [kg/m^3]",
                                "Pre-Shock Soundspeed [m/s]"])
        if check_flag.get('Post-Shock Thermodynamic State', False):
            header_data.extend(["Post-Shock Temperature [K]", "Post-Shock Pressure [Pa]", "Post-Shock Density [kg/m^3]",
                                "Post-Shock Soundspeed [m/s]"])
        if check_flag.get('Flame Surface Length', False):
            header_data.extend(["Flame Surface Length [m]"])
        # Step 1: Write the header portion of the results file
        if smooth_check:
            file_name = "Wave-Tracking-Smooth-Results.txt"
        else:
            file_name = "Wave-Tracking-Results.txt"

        with open(os.path.join(file_path, file_name), "w") as outfile:
            outfile.write("#")
            for i in range(len(header_data)):
                outfile.write("{0:<55s} ".format(header_data[i]))
            outfile.write("\n")
            # Step 2:
            for i in range(len(position_data)):  # Number of Time indices
                outfile.write(" {0:<55e}".format(position_data[i][0]))
                if check_flag.get('Position Flag', False):
                    outfile.write(" {0:<55e} {1:<55e} {2:<55e}".format(position_data[i][1],
                                                                       position_data[i][2],
                                                                       position_data[i][3]))
                if check_flag.get('Lab Velocity Flag', False):
                    outfile.write(" {0:<55e} {1:<55e}".format(lab_ref_vel[0][i],
                                                              lab_ref_vel[1][i]))
                if check_flag.get('Relative Velocity Flag', False):
                    outfile.write(" {0:<55e} {1:<55e} {1:<55e}".format(rel_ref_vel[i],
                                                                       lab_ref_vel[0][i] - rel_ref_vel[i],
                                                                       lab_ref_vel[1][i] - rel_ref_vel[i]))
                if check_flag.get('Flame Thermodynamic State', False):
                    outfile.write(" {0:<55e} {1:<55e} {2:<55e} {3:<55e}".format(flame_thermo_data[i][0],
                                                                                flame_thermo_data[i][1],
                                                                                flame_thermo_data[i][2],
                                                                                flame_thermo_data[i][3]))
                if check_flag.get('Pre-Shock Thermodynamic State', False):
                    outfile.write(" {0:<55e} {1:<55e} {2:<55e} {3:<55e}".format(pre_shock_thermo_data[i][0],
                                                                                pre_shock_thermo_data[i][1],
                                                                                pre_shock_thermo_data[i][2],
                                                                                pre_shock_thermo_data[i][3]))
                if check_flag.get('Post-Shock Thermodynamic State', False):
                    outfile.write(" {0:<55e} {1:<55e} {2:<55e} {3:<55e}".format(post_shock_thermo_data[i][0],
                                                                                post_shock_thermo_data[i][1],
                                                                                post_shock_thermo_data[i][2],
                                                                                post_shock_thermo_data[i][3]))
                if check_flag.get('Flame Surface Length', False):
                    outfile.write(" {0:<55e}".format(position_data[i][0]))
                outfile.write("\n")
            outfile.close()
        return

    ####################################################################################################################
    #
    ####################################################################################################################
    # Script Processing Constants
    poly_fit_bin_size = 51
    poly_fit_order = 2

    # Step 0:

    # Step 0:
    temp_location_evolution = []
    raw_wave_velocity = []
    smooth_wave_velocity = []
    raw_gas_velocity_evolution = []
    smooth_gas_velocity_evolution = []
    raw_flame_thermo_data = []
    raw_pre_shock_thermo_data = []
    raw_post_shock_thermo_data = []
    smooth_thermodynamic_data = []
    raw_flame_area = []
    # Step 1: Determine the wave positions and velocities in the laboratory frame of reference
    if check_flag.get('Position Flag', False):
        temp_location_evolution = parallelWaveProcessingFunction(data_dir,
                                                                 row_index,
                                                                 wavePositionProcessingFunction, 16)
        # Separate out the individual wave position arrays
        raw_temp_array = np.empty((len(temp_location_evolution), 4), dtype=float)
        for i in range(len(temp_location_evolution)):
            raw_temp_array[i, 0] = temp_location_evolution[i][0]
            raw_temp_array[i, 1] = temp_location_evolution[i][1]
            raw_temp_array[i, 2] = temp_location_evolution[i][2]
            raw_temp_array[i, 3] = temp_location_evolution[i][3]
        # Get indices that would sort the first array
        sorted_indices = np.argsort(raw_temp_array[:, 0])
        # Apply sorting to all columns of raw_temp_array
        sorted_raw_temp_array = raw_temp_array[sorted_indices]

        # Evaluate the polynomial at a given point within a certain bin
        [time_array, flame_position, smooth_flame_vel] = polynomial_fit_over_array(sorted_raw_temp_array[:, 0],
                                                                                   sorted_raw_temp_array[:, 1],
                                                                                   poly_fit_bin_size, poly_fit_order)
        [_, shock_position, smooth_shock_vel] = polynomial_fit_over_array(sorted_raw_temp_array[:, 0],
                                                                          sorted_raw_temp_array[:, 2],
                                                                          poly_fit_bin_size, poly_fit_order)
        [_, max_position_position, _] = polynomial_fit_over_array(sorted_raw_temp_array[:, 0],
                                                                  sorted_raw_temp_array[:, 3],
                                                                  poly_fit_bin_size, poly_fit_order)

        smooth_possition_array = np.empty((len(time_array), 4), dtype=float)
        for i in range(len(time_array)):
            smooth_possition_array[i, 0] = time_array[i]
            smooth_possition_array[i, 1] = flame_position[i]
            smooth_possition_array[i, 2] = shock_position[i]
            smooth_possition_array[i, 3] = max_position_position[i]
    if check_flag.get('Lab Velocity Flag', False):
        raw_wave_velocity = np.empty(2, dtype=object)
        raw_wave_velocity[0] = np.gradient(sorted_raw_temp_array[:, 1]) / np.gradient(sorted_raw_temp_array[:, 0])
        raw_wave_velocity[1] = np.gradient(sorted_raw_temp_array[:, 2]) / np.gradient(sorted_raw_temp_array[:, 0])

        smooth_wave_velocity = np.empty(2, dtype=object)
        smooth_wave_velocity[0] = smooth_flame_vel
        smooth_wave_velocity[1] = smooth_shock_vel
    # Step 2: Determine the gas velocity and thermodynamic state from the simulation results
    if check_flag.get('Relative Velocity Flag', False):
        raw_gas_velocity_evolution = parallelRelativeProcessingFunction(data_dir,
                                                                        row_index,
                                                                        gasVelocityProcessingFunction, 6)
        [_, smooth_gas_velocity_evolution, _] = polynomial_fit_over_array(sorted_raw_temp_array[:, 0],
                                                                          raw_gas_velocity_evolution,
                                                                          poly_fit_bin_size, poly_fit_order)
    # Step 3:
    if check_flag.get('Flame Thermodynamic State', False):
        raw_flame_thermo_data = parallelThermodynamicProcessingFunction(data_dir,
                                                                        row_index,
                                                                        input_params,
                                                                        "flame",
                                                                        thermodynamicProcessingFunction, 6)
    if check_flag.get('Pre-Shock Thermodynamic State', False):
        raw_pre_shock_thermo_data = parallelThermodynamicProcessingFunction(data_dir,
                                                                            row_index,
                                                                            input_params,
                                                                            "pre-shock",
                                                                            thermodynamicProcessingFunction, 6)
    if check_flag.get('Post-Shock Thermodynamic State', False):
        raw_post_shock_thermo_data = parallelThermodynamicProcessingFunction(data_dir,
                                                                             row_index,
                                                                             input_params,
                                                                             "post-shock",
                                                                             thermodynamicProcessingFunction, 6)
    if check_flag.get('Flame Surface Length', False):
        raw_flame_area = parallelFlameAreaProcessingFunction(data_dir,
                                                             row_index,
                                                             flameAreaContourFunction,
                                                             6)
    # Step 4:

    writeWaveResults(output_dir, False, check_flag, sorted_raw_temp_array, raw_wave_velocity,
                     raw_gas_velocity_evolution,
                     raw_flame_thermo_data, raw_pre_shock_thermo_data, raw_post_shock_thermo_data)

    writeWaveResults(output_dir, True, check_flag, smooth_temp_array, smooth_wave_velocity,
                     smooth_gas_velocity_evolution,
                     smooth_thermodynamic_data)
    return


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

    def writeTextFile(file_path):
        """

        :param file_path:
        :return:
        """
        # Step 1: Dynamically create the text file header depending on the assigned flags
        header_data = ["Time [s]"]
        if check_flags.get('Position Flag', False):
            header_data.extend(["Flame Position [m]", "Shock Position [m]", "Max Pressure Position [m]"])
        if check_flags.get('Lab Velocity Flag', False):
            header_data.extend(["Flame Lab Velocity [m]", "Shock Lab Velocity [m]"])
        if check_flags.get('Relative Velocity Flag', False):
            header_data.extend(["Gas Velocity [m/s]", "Flame Ref Velocity [m]", "Shock Ref Velocity [m]"])
        if check_flags.get('Flame Thermodynamic State', False):
            header_data.extend(["Flame Temperature [K]", "Flame Pressure [Pa]", "Flame Density [kg/m^3]",
                                "Flame Soundspeed [m/s]"])
        if check_flags.get('Pre-Shock Thermodynamic State', False):
            header_data.extend(["Pre-Shock Temperature [K]", "Pre-Shock Pressure [Pa]", "Pre-Shock Density [kg/m^3]",
                                "Pre-Shock Soundspeed [m/s]"])
        if check_flags.get('Post-Shock Thermodynamic State', False):
            header_data.extend(["Post-Shock Temperature [K]", "Post-Shock Pressure [Pa]", "Post-Shock Density [kg/m^3]",
                                "Post-Shock Soundspeed [m/s]"])
        if check_flags.get('Flame Surface Length', False):
            header_data.extend(["Flame Surface Length [m]"])
        # Step 2:
        with open(file_path, "w") as outfile:
            outfile.write("#")
            for i in range(len(header_data)):
                outfile.write("{0:<55s} ".format(header_data[i]))
            outfile.write("\n")
            # Step 2:
            for i in range(len(position_data)):  # Number of Time indices
                outfile.write(" {0:<55e}".format(position_data[i][0]))
                if check_flags.get('Position Flag', False):
                    outfile.write(" {0:<55e} {1:<55e} {2:<55e}".format(position_data[i][1],
                                                                       position_data[i][2],
                                                                       position_data[i][3]))
                if check_flags.get('Lab Velocity Flag', False):
                    outfile.write(" {0:<55e} {1:<55e}".format(lab_ref_vel[0][i],
                                                              lab_ref_vel[1][i]))
                if check_flags.get('Relative Velocity Flag', False):
                    outfile.write(" {0:<55e} {1:<55e} {1:<55e}".format(rel_ref_vel[i],
                                                                       lab_ref_vel[0][i] - rel_ref_vel[i],
                                                                       lab_ref_vel[1][i] - rel_ref_vel[i]))
                if check_flags.get('Flame Thermodynamic State', False):
                    outfile.write(" {0:<55e} {1:<55e} {2:<55e} {3:<55e}".format(flame_thermo_data[i][0],
                                                                                flame_thermo_data[i][1],
                                                                                flame_thermo_data[i][2],
                                                                                flame_thermo_data[i][3]))
                if check_flags.get('Pre-Shock Thermodynamic State', False):
                    outfile.write(" {0:<55e} {1:<55e} {2:<55e} {3:<55e}".format(pre_shock_thermo_data[i][0],
                                                                                pre_shock_thermo_data[i][1],
                                                                                pre_shock_thermo_data[i][2],
                                                                                pre_shock_thermo_data[i][3]))
                if check_flags.get('Post-Shock Thermodynamic State', False):
                    outfile.write(" {0:<55e} {1:<55e} {2:<55e} {3:<55e}".format(post_shock_thermo_data[i][0],
                                                                                post_shock_thermo_data[i][1],
                                                                                post_shock_thermo_data[i][2],
                                                                                post_shock_thermo_data[i][3]))
                if check_flags.get('Flame Surface Length', False):
                    outfile.write(" {0:<55e}".format(position_data[i][0]))
                outfile.write("\n")
            outfile.close()
        return

    """

    """
    # Step 0:

    # Step 1:

    # Step 2:
    # Step 2.1: Flame Processing
    if 'Flame Processing' in check_flags:
        if check_flags['Flame Processing'].get('Position', False):

        if check_flags['Flame Processing'].get('Velocity', False):
            if check_flags['Flame Processing'].get('Position', True):
                print('Error: Must '

                if check_flags['Flame Processing'].get('Relative Velocity', False):
                    if
                check_flags['Flame Processing'].get('Thermodynamic State', False):

                if check_flags['Flame Processing'].get('Surface Length', False):

                # Step 2.2: Maximum Pressure Processing

                # Step 2.3: Pre-Shock Processing

                # Step 2.4: Post-Shock Processing

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
            'Position': False,
            'Velocity': False,
            'Relative Velocity': False,
            'Thermodynamic State': False,
            'Surface Length': True
        },
        'Maximum Pressure Processing': {
            'Position': False,
            'Velocity': False,
            'Thermodynamic State': False
        },
        'Pre-Shock Processing': {
            'Position': False,
            'Velocity': False,
            'Thermodynamic State': False
        },
        'Post-Shock Processing': {
            'Position': False,
            'Velocity': False,
            'Thermodynamic State': False
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
    waveProcessingFunction(updated_data_list, domain_sizing, input_params, check_flag=check_flag_dict,
                           output_dir=output_dir_path)

    return


if __name__ == "__main__":
    main()