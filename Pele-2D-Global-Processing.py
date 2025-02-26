import os, re, yt, cv2, itertools, multiprocessing
import numpy as np
import cantera as ct
from ast import literal_eval
from itertools import groupby
import matplotlib.pyplot as plt

class MyClass():
    pass
def importRawData(fileName):
    with open(fileName, 'r') as f:
        next(f); next(f); next(f)
        dataFile = (k.split() for k in f.read().splitlines())

    data = []
    for _, v in groupby(dataFile, lambda x: x != []):
        bb = list(v)
        if bb != [[],[]]:
            data.append([literal_eval(str(k)) for k in bb])
    return data

def importGlobalData(fileName):
    # Step 0: Load the wave tracking data for time (0), flame position (1), shock position (2), and pressure wave (3)
    data = np.loadtxt(fileName, encoding="utf8", unpack=True)
    return data[0,:], data[1,:], data[2,:], data[3,:]


def wrtieWaveData(file_path, data):
    # Step 1: Write the header portion of the results file
    with open(f"{file_path}\\Wave-Tracking-Results.txt", "w") as outfile:
        outfile.write("#{0:<55s} {1:<55s} {2:<55s} \n".format("Time [s]", "Flame Position [m]", "Shock Position [m]", "Max Pressure Position [m]"))
        # Step 2:
        for i in range(len(data)):  # Number of Time indices
            outfile.write(" {0:<55e} {1:<55e} {2:<55e} {3:<55e}\n".format(data[i][0], data[i][1], data[i][2], data[i][3]))
        outfile.close
    return

def writeGlobalData(file_path, data, y_position):
    # Step 1: Write the header portion of the results file
    with open(f"{file_path}\\Truncated-Simulation-Results.txt", "w") as outfile:
        if isinstance(y_position, str) is True:
            outfile.write("# Y Position {0:<.0s} \n".format(y_position))
        else:
            outfile.write("# Y Position {0:<.0f} \n".format(y_position))
        for i in range(len(data)):  # Number of Time indices
            outfile.write("# {0:<.0f} - Time = {1:<55e} \n".format(i, float(data[i][0])))
            for j in range(len(data[i][1])):
                outfile.write("     {0:<55e} {1:<55e} {2:<55e} {3:<55e} {4:<55e} {5:<55e} {6:<55e} {7:<55e} \n".format(float(data[i][1][j]),
                                                  float(data[i][2][j]),
                                                  float(data[i][3][j]),
                                                  float(data[i][4][j]),
                                                  float(data[i][5][j]),
                                                  float(data[i][6][j]),
                                                  float(data[i][7][j]),
                                                  float(data[i][8][j])))
            outfile.write('\n\n')
        outfile.close()
    return

def waveProcessingFunction(args):
    def waveTrackingFunction(data, y_var, tracking_str, tracking_var, wave_type):
        # Step 1: Load the desired marker and x positions
        temp_data = data["boxlib", str(tracking_str)]
        temp_x_pos = data["boxlib", str('x')]
        # Step 2:
        if wave_type == "Flame":
            wave_index = np.argwhere(temp_data[:, y_var, 0] >= tracking_var)[-1][0]
        elif wave_type == "Shock":
            tracking_gradient = abs(np.gradient(temp_data[:, y_var, 0].to_value()) / np.gradient(temp_x_pos[:, y_var, 0].to_value()))
            wave_index = np.argwhere(tracking_gradient == np.max(tracking_gradient))[-1][0]
        elif wave_type == "Max Pressure":
            temp_index = temp_data[:, y_var, 0].to_value()
            wave_index = np.argwhere(temp_index == np.max(temp_index))[-1][0]
        return temp_x_pos[wave_index, y_var, 0].to_value() / 100
    #####
    #
    #####
    # Step 0: Unpack the parallelization arguments
    current_file = args[0]
    input_params = args[1]
    y_location = args[2]
    # Step 1: Determine the file path for the highest level
    level_var = -1
    for level in os.listdir(current_file):
        if os.path.isdir(os.path.join(current_file, level)):
            if int(level.split('_')[-1]) > level_var:
                level_var = int(level.split('_')[-1])

    raw_data = yt.load(current_file)
    time = raw_data.current_time.to_value()
    imported_data = raw_data.covering_grid(level_var,
                                           left_edge=[0.0, 0.0, 0.0],
                                           dims=raw_data.domain_dimensions * [2 ** level_var, 2 ** level_var, 1],
                                           # And any fields to preload (this is optional!)
                                           # fields=desired_varables
                                           )
    # Step 2:
    if isinstance(y_location, str) is True:
        if y_location == "Bottom":
            y_index = 0
        elif y_location == "Top":
            y_index = len(imported_data["boxlib", str('y')][1]) - 1
        else:
            y_index = int(len(imported_data["boxlib", str('y')][1]) / 2 - 1)
    else:
        y_index = y_location
    # Step 3:
    flame_position = waveTrackingFunction(imported_data, y_index, "Temp", 2000, "Flame")
    shock_position = waveTrackingFunction(imported_data, y_index, "pressure", None, "Shock")
    max_pressure_position = waveTrackingFunction(imported_data, y_index, "pressure", None, "Max Pressure")
    wave_array = np.array([time, flame_position, shock_position, max_pressure_position])
    return wave_array

def globalProcessingFunction(args):
    #####
    #
    #####
    file_path = args[0]
    input_params = args[1]
    y_location = args[2]
    desired_variables = args[3]
    # Step 1: Determine the file path for the highest level
    level_var = -1
    for level in os.listdir(file_path):
        if os.path.isdir(os.path.join(file_path, level)):
            if int(level.split('_')[-1]) > level_var:
                level_var = int(level.split('_')[-1])

    raw_data = yt.load(file_path)
    imported_data = raw_data.covering_grid(level_var,
                                           left_edge=[0.0, 0.0, 0.0],
                                           dims=raw_data.domain_dimensions * [2 ** level_var, 2 ** level_var, 1],
                                           # And any fields to preload (this is optional!)
                                           # fields=desired_varables
                                           )
    # Step 2:
    if isinstance(y_location, str) is True:
        if y_location == "Bottom":
            y_index = 0
        elif y_location == "Top":
            y_index = len(imported_data["boxlib", str('y')][1]) - 1
        else:
            y_index = int(len(imported_data["boxlib", str('y')][1]) / 2 - 1)
    else:
        y_index = y_location
    # Step 2:
    data = np.empty(len(desired_variables)+1, dtype=object)
    for i in range(len(desired_variables)):
        # Step 2.1:
        temp_data = imported_data["boxlib", str(desired_variables[i])]
        var_array_units = temp_data[:, y_index, 0]
        # Step 2.2:
        if i == 0:
            data[i] = raw_data.current_time.to_value()
        else:
            if desired_variables[i] == "x" or desired_variables[i] == "x_velocity" or desired_variables[i] == "y_velocity":
                data[i] = var_array_units.to_value() / 100  # desired value is stored in cm units
            elif desired_variables[i] == "pressure":
                data[i] = var_array_units.to_value() / 10  # desired value is stored in cm units
            else:
                data[i] = var_array_units.to_value()
    return data

def ParallelWaveProcessingFunction(paralellizationList, input_parameters, y_location, predicate, nProcs):
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
                         itertools.repeat(input_parameters),
                         itertools.repeat(y_location)))
    return y

def ParallelGlobalProcessingFunction(paralellizationList, input_parameters, y_location, desired_variables, predicate, nProcs):
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
                         itertools.repeat(input_parameters),
                         itertools.repeat(y_location),
                         itertools.repeat(desired_variables)))
    return y

def main():
    ####################################################################################################################
    # This code is developed to process a 2D Planar Flame simulated using PeleC for a given y-position and a given
    # (temperature) isotherm for the flame and pressure for any shock
    #
    # All functions are configured for a 2 dimensional space
    ####################################################################################################################
    # Step 1: Initialize the code with the desired processed variables and mixture composition
    skip_load = 10
    y_location = "Middle"
    desired_variables = np.array(["x", "Temp", "pressure", "x_velocity", "y_velocity", "Y(H2)", "Y(OH)", "Y(HO2)", "Y(H)"])

    input_params = MyClass()
    input_params.T = 298.15
    input_params.P = 1.0 * ct.one_atm
    input_params.Phi = 1.0
    input_params.Fuel = 'H2'

    if input_params.Fuel == "H2":
        input_params.oxygenAmount = 0.5
    if input_params.Fuel == "C2H6":
        input_params.oxygenAmount = 3.5
    if input_params.Fuel == "C4H10":
        input_params.oxygenAmount = 6.5
    input_params.nitrogenAmount = 0
    input_params.X = {input_params.Fuel: input_params.Phi, 'O2': input_params.oxygenAmount,
                      'N2': input_params.nitrogenAmount}
    input_params.mech = 'LiDryer-mechanism.yaml'
    # Step 2: Create the result directories
    dir_path = os.path.dirname(os.path.realpath(__file__))
    if os.path.exists(f"{dir_path}\\Processed-Global-Results\\") is False:
        os.mkdir(f"{dir_path}\\Processed-Global-Results\\")
    output_dir_path = f"{dir_path}\\Processed-Global-Results\\"
    # Step 3: Collect all the present pelec data directories
    time_data_dir = []
    for raw_data_folder in os.listdir(dir_path):
        if os.path.isdir(os.path.join(dir_path, raw_data_folder)) and raw_data_folder.startswith('Raw-PeleC'):
            raw_data_dir = f"{dir_path}\\{raw_data_folder}\\"
            for time_step in os.listdir(raw_data_dir):
                if os.path.isdir(os.path.join(raw_data_dir, time_step)) and time_step.startswith('plt'):
                    time_data_dir.append(os.path.join(raw_data_dir, time_step))
    if skip_load > 0:
        time_data_dir = time_data_dir[0::skip_load]
    # Step 4: Check for prior processed data
    """
    for raw_data_folder in os.listdir(dir_path):
        if os.path.isdir(os.path.join(dir_path, raw_data_folder)) and raw_data_folder.startswith('Processed-Global-Results'):
            for global_files in os.listdir(os.path.join(dir_path, raw_data_folder)):
                if global_files == "Truncated-Simulation-Results.txt":
                    truncated_results = importRawData(global_files)
                if global_files == "Wave-Tracking-Results.txt":
                    wave_data = global_files
    """
    # Step 5:
    flame_location_evolution = ParallelWaveProcessingFunction(time_data_dir,
                                                              input_params,
                                                              y_location,
                                                              waveProcessingFunction, 6)
    wrtieWaveData(output_dir_path, flame_location_evolution)
    # Step 6:
    """
    global_evolution = ParallelGlobalProcessingFunction(time_data_dir,
                                                        input_params,
                                                        y_location,
                                                        desired_variables,
                                                        globalProcessingFunction, 6)
    writeGlobalData(output_dir_path, global_evolution, y_location)
    """
    # Step 7:


    return

if __name__ == "__main__":
    main()