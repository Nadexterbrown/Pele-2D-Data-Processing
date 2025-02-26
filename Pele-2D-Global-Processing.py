import os, re, yt, cv2, itertools, multiprocessing
import numpy as np
import cantera as ct
from ast import literal_eval
from itertools import groupby
import matplotlib.pyplot as plt
from sdtoolbox.thermo import soundspeed_fr
from sdtoolbox.postshock import PostShock_fr


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


def processing_height(directory_path, desired_y_location):
    # Step 1:
    ds = yt.load(directory_path)

    # Step 2: Load the data for physical size extraction
    data = ds.covering_grid(0,
                            left_edge=[0.0, 0.0, 0.0],
                            dims=ds.domain_dimensions * [2 ** 0, 2 ** 0, 1],
                            # And any fields to preload (this is optional!)
                            # fields=desired_varables
                            )

    # Step 3:
    if isinstance(desired_y_location, str) is True:
        if desired_y_location == "Bottom":
            y_index = 0
        elif desired_y_location == "Top":
            y_index = int(ds.domain_dimensions[1] - 1)
        else:
            y_index = int((ds.domain_dimensions[1] / 2) - 1)
    else:
        temp_array = np.argwhere(data["boxlib", str('y')][0][:].to_value() <= desired_y_location)
        y_index = temp_array[-1][0]
    return np.array([data["boxlib", str('x')][-1][0].to_value()[0], data["boxlib", str('y')][0][y_index].to_value()[0]])


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
    domain_size = args[1]
    # Step 1: Determine the file path for the highest level
    level_var = -1
    for level in os.listdir(current_file):
        if os.path.isdir(os.path.join(current_file, level)):
            if int(level.split('_')[-1]) > level_var:
                level_var = int(level.split('_')[-1])

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
    domain_size = args[1]
    # Step 1: Determine the file path for the highest level
    level_var = -1
    for level in os.listdir(current_file):
        if os.path.isdir(os.path.join(current_file, level)):
            if int(level.split('_')[-1]) > level_var:
                level_var = int(level.split('_')[-1])

    raw_data = yt.load(current_file)
    slice = raw_data.ray(np.array([0.0, domain_size[1], 0.0]), np.array([domain_size[0], domain_size[1],
                                                                         0.0]))  # 1 corresponds to the y-axis, y_location is the physical position (in meters)
    ray_sort = np.argsort(slice["x"])
    time = raw_data.current_time.to_value()
    # Step 2:
    gas_vel = preheatZoneFunction(slice, ray_sort)
    return gas_vel


def thermodynamicProcessingFunction(args):
    def thermodynamicParameters(data, ray_sort, input_params):
        # Step 0:
        [shock_index, _] = waveTrackingFunction(data, ray_sort, "pressure", None, "Shock")
        # Step 1:
        temp_temp = data["boxlib", "Temp"][ray_sort].to_value()
        temp_pres = data["boxlib", "pressure"][ray_sort].to_value()

        temp_comp = []
        for i in range(len(input_params.result_species)):
            temp_comp.append(data["boxlib", str("Y(" + input_params.result_species[i] + ")")][ray_sort].to_value())

        mixture_comp = {}
        for i in range(len(input_params.result_species)):
            mixture_comp.update({str(input_params.result_species[i]): temp_comp[i][shock_index - 10]})

        # Step 0: Allocate the space for all thermodynamic parameters (reactant temperature, pressure, density, soundspeed
        # and the coresponding product values)
        result_array = np.empty(2, dtype=float)
        # Step 1: Using the input parameter class, create a cantera gas object
        gas_temp = ct.Solution(input_params.mech)
        gas_temp.TPY = (temp_temp[shock_index - 10],
                        temp_pres[shock_index - 10] * 10,
                        mixture_comp)
        result_array[0] = gas_temp.density_mass
        result_array[1] = soundspeed_fr(gas_temp)
        del gas_temp
        return result_array

    #####
    #
    #####
    # Step 1:
    current_file = args[0]
    domain_size = args[1]
    input_params = args[2]
    # Step 2: Determine the file path for the highest level
    level_var = -1
    for level in os.listdir(current_file):
        if os.path.isdir(os.path.join(current_file, level)):
            if int(level.split('_')[-1]) > level_var:
                level_var = int(level.split('_')[-1])

    raw_data = yt.load(current_file)
    slice = raw_data.ray(np.array([0.0, domain_size[1], 0.0]), np.array([domain_size[0], domain_size[1],
                                                                         0.0]))  # 1 corresponds to the y-axis, y_location is the physical position (in meters)
    ray_sort = np.argsort(slice["x"])
    time = raw_data.current_time.to_value()
    # Step 4:
    thermo_results = thermodynamicParameters(slice, ray_sort, input_params)
    return thermo_results


def waveProcessingFunction(data_dir, row_index, input_params, position_check=True,
                           lab_vel_check=False,
                           rel_vel_check=False,
                           thermo_check=None,
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

    def parallelThermodynamicProcessingFunction(paralellizationList, y_location, input_parameters, predicate, nProcs):
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
                             itertools.repeat(input_parameters)))
        return y

    def writeWaveResults(file_path, position_data, lab_ref_vel, rel_ref_vel, thermo_data, smooth_check, position_check,
                         lab_vel_check, rel_vel_check, thermo_check):
        # Step 0:
        header_data = ["Time [s]"]
        if position_check:
            header_data.extend(["Flame Position [m]", "Shock Position [m]", "Max Pressure Position [m]"])
        if lab_vel_check:
            header_data.extend(["Flame Lab Velocity [m]", "Shock Lab Velocity [m]"])
        if rel_vel_check:
            header_data.extend(["Gas Velocity [m/s]", "Flame Ref Velocity [m]", "Shock Ref Velocity [m]"])
        if thermo_check:
            header_data.extend(["Flame Density [kg/m^3]", "Flame Soundspeed [m/s]"])
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
                if position_check:
                    outfile.write(" {0:<55e} {1:<55e} {2:<55e}".format(position_data[i][1],
                                                                       position_data[i][2],
                                                                       position_data[i][3]))
                if lab_vel_check:
                    outfile.write(" {0:<55e} {1:<55e}".format(lab_ref_vel[0][i],
                                                              lab_ref_vel[1][i]))
                if rel_vel_check:
                    outfile.write(" {0:<55e} {1:<55e} {1:<55e}".format(rel_ref_vel[i],
                                                                       lab_ref_vel[0][i] - rel_ref_vel[i],
                                                                       lab_ref_vel[1][i] - rel_ref_vel[i]))
                if thermo_check:
                    outfile.write(" {0:<55e} {1:<55e}".format(thermo_data[i][0],
                                                              thermo_data[i][1]))
                outfile.write("\n")
            outfile.close()
        return

    ####################################################################################################################
    #
    ####################################################################################################################
    poly_fit_bin_size = 51
    poly_fit_order = 2

    # Step 0:
    temp_location_evolution = []
    raw_wave_velocity = []
    smooth_wave_velocity = []
    raw_gas_velocity_evolution = []
    smooth_gas_velocity_evolution = []
    raw_thermodynamic_data = []
    smooth_thermodynamic_data = []
    # Step 1: Determine the wave positions and velocities in the laboratory frame of reference
    if position_check:
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
        [_, max_position_position, _] = polynomial_fit_over_array(sorted_raw_temp_array[:, 0], sorted_raw_temp_array[:, 3],
                                                               poly_fit_bin_size, poly_fit_order)

        smooth_temp_array = np.empty((len(time_array), 4), dtype=float)
        for i in range(len(time_array)):
            smooth_temp_array[i, 0] = time_array[i]
            smooth_temp_array[i, 1] = flame_position[i]
            smooth_temp_array[i, 2] = shock_position[i]
            smooth_temp_array[i, 3] = max_position_position[i]

    if lab_vel_check:
        raw_wave_velocity = np.empty(2, dtype=object)
        raw_wave_velocity[0] = np.gradient(sorted_raw_temp_array[:, 1]) / np.gradient(sorted_raw_temp_array[:, 0])
        raw_wave_velocity[1] = np.gradient(sorted_raw_temp_array[:, 2]) / np.gradient(sorted_raw_temp_array[:, 0])

        smooth_wave_velocity = np.empty(2, dtype=object)
        smooth_wave_velocity[0] = smooth_flame_vel
        smooth_wave_velocity[1] = smooth_shock_vel

    # Step 2: Determine the gas velocity and thermodynamic state from the simulation results
    if rel_vel_check:
        raw_gas_velocity_evolution = parallelRelativeProcessingFunction(data_dir,
                                                                        row_index,
                                                                        gasVelocityProcessingFunction, 6)
        [_, smooth_gas_velocity_evolution, _] = polynomial_fit_over_array(sorted_raw_temp_array[:, 0],
                                                                       raw_gas_velocity_evolution,
                                                                       poly_fit_bin_size, poly_fit_order)
    # Step 3:
    if thermo_check:
        raw_thermodynamic_data = parallelThermodynamicProcessingFunction(data_dir,
                                                                         row_index,
                                                                         input_params,
                                                                         thermodynamicProcessingFunction, 6)
        [_, smooth_thermodynamic_data, _] = polynomial_fit_over_array(sorted_raw_temp_array[:, 0], raw_thermodynamic_data,
                                                                   poly_fit_bin_size, poly_fit_order)
    # Step 4:
    writeWaveResults(output_dir, sorted_raw_temp_array, raw_wave_velocity, raw_gas_velocity_evolution,
                     raw_thermodynamic_data, False, position_check, lab_vel_check, rel_vel_check,
                     thermo_check)

    writeWaveResults(output_dir, smooth_temp_array, smooth_wave_velocity, smooth_gas_velocity_evolution,
                     smooth_thermodynamic_data, True, position_check, lab_vel_check, rel_vel_check,
                     thermo_check)
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
    raw_data_truncation = False
    wave_position = True
    wave_lab_velocity = True
    wave_rel_velocity = False
    thermodynamic_state_check = False

    skip_load = 0  # 0 for no skip
    # row_index = "Middle"  # Desired row location for data collection
    row_index = 0.1 # Desired y_location for data collection in cm
    mass_fraction_variables = np.array(["H", "H2", "H2O", "H2O2", "HO2", "N2", "O", "O2", "OH"])

    # Step 1: Initialize the code with the desired processed variables and mixture composition
    input_params = MyClass()
    input_params.T = 298.15
    input_params.P = 1.0 * ct.one_atm
    input_params.Phi = 1.5
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
    updated_data_list = reload_prior_processed_data(time_data_dir, os.path.join(output_dir_path, f"Wave-Tracking-Results.txt"),
                                                    check=reload_data)
    # Step 5: Truncate the raw data list if skip loading is enabled
    if skip_load > 0:
        updated_data_list = updated_data_list[0::skip_load]
    # Step 5:
    domain_size = processing_height(updated_data_list[0], row_index)

    # Step 6:
    waveProcessingFunction(updated_data_list, domain_size, input_params, position_check=wave_position,
                           lab_vel_check=wave_lab_velocity,
                           rel_vel_check=wave_rel_velocity,
                           thermo_check=thermodynamic_state_check,
                           output_dir=output_dir_path)

    return


if __name__ == "__main__":
    main()