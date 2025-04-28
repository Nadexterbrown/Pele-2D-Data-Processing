import os
import yt
import time
import numpy as np
from mpi4py import MPI
from typing import Optional
from dataclasses import dataclass, field, fields

from utilities.general import *
from utilities.pele import *

yt.set_log_level(0)

########################################################################################################################
# Global Program Setting Variables
########################################################################################################################

version = "1.0.0"

flame_temp = 2500.0  # Temperature for flame contour extraction

########################################################################################################################
# Script Configuration Classes
########################################################################################################################

@dataclass
class DataExtractionConfig:
    Parallelize: bool = False
    Location: Optional[float] = None
    Grid: Optional[np.ndarray] = None


@dataclass
class FieldConfig:
    Name: Optional[str]
    Flag: Optional[bool] = False
    Offset: Optional[float] = 0.0
    Animation: Optional[bool] = False
    Local: Optional[bool] = False
    Collective: Optional[bool] = False


@dataclass
class FlameConfig:
    Position: FieldConfig = field(default_factory=lambda: FieldConfig(Name='Flame Position', Flag=True))
    Velocity: FieldConfig = field(default_factory=lambda: FieldConfig(Name='Flame Velocity', Flag=True))
    RelativeVelocity: FieldConfig = field(default_factory=lambda: FieldConfig(Name='Flame Relative Velocity', Flag=True))
    ThermodynamicState: FieldConfig = field(default_factory=lambda: FieldConfig(Name='Flame Thermodynamic State', Flag=True, Offset=10e-6))
    HeatReleaseRate: FieldConfig = field(default_factory=lambda: FieldConfig(Name='Heat Release Rate', Flag=True))
    FlameThickness: FieldConfig = field(default_factory=lambda: FieldConfig(Name='Flame Thickness', Flag=True))  # fixed typo here too
    SurfaceLength: FieldConfig = field(default_factory=lambda: FieldConfig(Name='Surface Length', Flag=True))
    ReynoldsNumber: FieldConfig = field(default_factory=lambda: FieldConfig(Name='Reynolds Number', Flag=True))


@dataclass
class BurnedGasConfig:
    GasVelocity: FieldConfig = field(default_factory=lambda: FieldConfig(Name='Gas Velocity', Flag=True, Offset=10e-6))
    ThermodynamicState: FieldConfig = field(default_factory=lambda: FieldConfig(Name='Flame Thermodynamic State', Flag=True, Offset=10e-6))


@dataclass
class ShockConfig:
    Position: FieldConfig = field(default_factory=lambda: FieldConfig(Name='Shock Position', Flag=True))
    Velocity: FieldConfig = field(default_factory=lambda: FieldConfig(Name='Shock Velocity', Flag=True))
    PreShockThermodynamicState: FieldConfig = field(default_factory=lambda: FieldConfig(Name='Pre-Shock Thermodynamic State', Flag=True, Offset=10e-6))
    PostShockThermodynamicState: FieldConfig = field(default_factory=lambda: FieldConfig(Name='Post-Shock Thermodynamic State', Flag=True, Offset=-10e-6))


@dataclass
class AnimationConfig:
    LocalWindowSize: FieldConfig = field(default_factory=lambda: FieldConfig(Name='Local Window Size', Offset=1e-3))
    Temperature: FieldConfig = field(default_factory=lambda: FieldConfig(Name='Temperature', Flag=True))
    Pressure: FieldConfig = field(default_factory=lambda: FieldConfig(Name='Pressure', Flag=True))
    GasVelocity: FieldConfig = field(default_factory=lambda: FieldConfig(Name='X Velocity', Flag=True))
    HeatReleaseRate: FieldConfig = field(default_factory=lambda: FieldConfig(Name='Heat Release Rate', Flag=True))
    FlameGeometry: FieldConfig = field(default_factory=lambda: FieldConfig(Name='Flame Geometry', Flag=True))
    FlameThickness: FieldConfig = field(default_factory=lambda: FieldConfig(Name='Flame Thickness', Flag=True))
    Schlieren: FieldConfig = field(default_factory=lambda: FieldConfig(Name='Schlieren', Flag=True))
    StreamLines: FieldConfig = field(default_factory=lambda: FieldConfig(Name='StreamLines', Flag=True))


@dataclass
class ScriptConfig:
    DataExtraction: DataExtractionConfig = field(default_factory=DataExtractionConfig)
    Flame: FlameConfig = field(default_factory=FlameConfig)
    BurnedGas: BurnedGasConfig = field(default_factory=BurnedGasConfig)
    Shock: ShockConfig = field(default_factory=ShockConfig)
    Animation: AnimationConfig = field(default_factory=AnimationConfig)


########################################################################################################################
# Specialized Pele Functions
########################################################################################################################

def single_file_processing(dataset, args, comm, logger):
    # Step 0: Collect current MPI rank and size
    rank = comm.Get_rank()
    size = comm.Get_size()

    ###############################################################################
    # Main Function
    ###############################################################################
    # Step 1: Unpack the arguments
    output_dir, script_config = args
    pltFile_name = dataset.basename
    extract_location = script_config.DataExtraction.Location + (dataset.index.get_smallest_dx().to_value() / 2)

    ###########################################
    # MPI Parallel Processing: Shared loading
    ###########################################
    # Step 2: Extract the data
    if rank == 0:
        append_to_log_file(logger, f"==== Processing {pltFile_name} ====")
        append_to_log_file(logger, f"   Loading {pltFile_name}...")

    time = dataset.current_time.to_value()
    data = data_ray_extraction(dataset, extract_location, comm, logger)
    # data = data_extraction(dataset, extract_location, comm, logger)

    if rank == 0:
        append_to_log_file(logger, f"   ...Done")

    # Step 3: Extract and calculate the flame geometry parameters taking advantage of yt parallelization with MPI

    if script_config.Flame.FlameThickness.Flag or script_config.Flame.SurfaceLength.Flag:
        if rank == 0:
            append_to_log_file(logger, f"       Flame Geometry Extraction...")
        tmp_dict = flame_geometry(dataset, output_dir, script_config, comm, logger)
        if rank == 0:
            append_to_log_file(logger, f"           ...Done")

    ###########################################
    # Serial Processing: Only root does heavy post-processing
    ###########################################
    if rank == 0:

        # Step 3: Process the data
        result_dict = {}
        result_dict['Time'] = time

        # Flame Processing
        if script_config.Flame.Position.Flag:
            append_to_log_file(logger, f"   Processing Flame...")

            result_dict['Flame'] = {}
            # Position
            if script_config.Flame.Position.Flag:
                append_to_log_file(logger, f"       Flame Position Extraction...")
                result_dict['Flame']['Index'], result_dict['Flame']['Position'] = wave_tracking('Flame',
                                                                                                pre_loaded_data=data)

                append_to_log_file(logger, f"           ...Done")

            # Gas Velocity
            if script_config.Flame.RelativeVelocity.Flag:
                append_to_log_file(logger, f"       Flame Gas Velocity Extraction...")
                tmp_idx = np.argmin(
                    abs(data['X'] - (result_dict['Flame']['Position'] + script_config.Flame.RelativeVelocity.Offset)))
                result_dict['Flame']['Gas Velocity'] = data['X Velocity'][tmp_idx]
                append_to_log_file(logger, f"           ...Done")

            # Thermodynamic State
            if script_config.Flame.ThermodynamicState.Flag:
                append_to_log_file(logger, f"       Flame Thermodynamic State Extraction...")
                result_dict['Flame']['Thermodynamic'] = thermodynamic_state_extractor(data,
                                                                                      result_dict['Flame']['Position'],
                                                                                      script_config.Flame.ThermodynamicState.Offset)
                append_to_log_file(logger, f"           ...Done")
            # Heat Release Rate
            if script_config.Flame.HeatReleaseRate.Flag:
                append_to_log_file(logger, f"       Flame Heat Release Extraction...")
                result_dict['Flame']['HRR'] = data['Heat Release Rate'][result_dict['Flame']['Index']]
                append_to_log_file(logger, f"           ...Done")
            # Flame Thickness
            if script_config.Flame.FlameThickness.Flag:
                result_dict['Flame']['Flame Thickness'] = tmp_dict['Flame Thickness']
            # Surface Length
            if script_config.Flame.SurfaceLength.Flag:
                result_dict['Flame']['Surface Length'] = tmp_dict['Flame Thickness']
            # Reynolds Number
            if script_config.Flame.ReynoldsNumber.Flag:
                append_to_log_file(logger, f"       Flame Reynolds Number Extraction...")
                result_dict['Flame']['Reynolds Number'] = reynolds_number(data)
                append_to_log_file(logger, f"           ...Done")

        # Burned Gas Processing
        if script_config.Flame.Position.Flag and (
                script_config.BurnedGas.GasVelocity.Flag or script_config.BurnedGas.ThermodynamicState.Flag):
            append_to_log_file(logger, f"   Processing Burned Gas...")

            result_dict['Burned Gas'] = {}
            if script_config.BurnedGas.GasVelocity.Flag:
                append_to_log_file(logger, f"       Burned Gas Velocity Extraction...")
                tmp_idx = np.argmin(
                    abs(data['X'] - (result_dict['Flame']['Position'] - script_config.BurnedGas.GasVelocity.Offset)))
                result_dict['Burned Gas']['Gas Velocity'] = data['X Velocity'][tmp_idx]
                append_to_log_file(logger, f"           ...Done")
            if script_config.BurnedGas.ThermodynamicState.Flag:
                append_to_log_file(logger, f"       Burned Gas Thermodynamic State Extraction...")
                result_dict['Burned Gas']['Thermodynamic'] = thermodynamic_state_extractor(data,
                                                                                           result_dict['Flame'][
                                                                                               'Position'],
                                                                                           script_config.BurnedGas.ThermodynamicState.Offset)
                append_to_log_file(logger, f"           ...Done")

        # Shock Processing
        if script_config.Shock.Position.Flag:
            append_to_log_file(logger, f"   Processing Shock...")

            result_dict['Shock'] = {}
            # Position
            if script_config.Shock.Position.Flag:
                append_to_log_file(logger, f"       Shock Position Extraction...")
                result_dict['Shock']['Index'], result_dict['Shock']['Position'] = wave_tracking('Shock',
                                                                                                pre_loaded_data=data)
                append_to_log_file(logger, f"           ...Done")

            if script_config.Shock.PreShockThermodynamicState.Flag:
                append_to_log_file(logger, f"       Pre-Shock Thermodynamic State Extraction...")
                result_dict['Shock']['PreShockThermodynamicState'] = thermodynamic_state_extractor(data,
                                                                                                   result_dict['Shock'][
                                                                                                       'Position'],
                                                                                                   script_config.Shock.PreShockThermodynamicState.Offset)
                append_to_log_file(logger, f"           ...Done")

            if script_config.Shock.PostShockThermodynamicState.Flag:
                append_to_log_file(logger, f"       Pre-Shock Thermodynamic State Extraction...")
                result_dict['Shock']['PostShockThermodynamicState'] = thermodynamic_state_extractor(data,
                                                                                                    result_dict[
                                                                                                        'Shock'][
                                                                                                        'Position'],
                                                                                                    script_config.Shock.PostShockThermodynamicState.Offset)
                append_to_log_file(logger, f"           ...Done")

        # Step 4: Create plot files
        append_to_log_file(logger, f"   Creating Single Animation Frames...")
        for field_info in fields(script_config.Animation):
            key = field_info.name
            val = getattr(script_config.Animation, key)

            if key in ('FlameGeometry', 'FlameThickness', 'Schlieren', 'StreamLines'):
                continue

            if val.Flag:
                tmp_dir = ensure_long_path_prefix(os.path.join(output_dir, "Animation-Frames", f"{key}-Plt-Files"))
                os.makedirs(tmp_dir, exist_ok=True)

                filename = os.path.join(tmp_dir, f"{pltFile_name}.png")
                animation_frame_generation(data['X'], data[val.Name], key, filename)
                if val.Local:
                    tmp_dir = ensure_long_path_prefix(
                        os.path.join(output_dir, "Animation-Frames", f"Local-{key}-Plt-Files"))
                    os.makedirs(tmp_dir, exist_ok=True)

                    filename = os.path.join(tmp_dir, f"{pltFile_name}.png")
                    animation_frame_generation(data['X'], data[val.Name], key, filename,
                                               plot_center=result_dict['Flame']['Position'],
                                               window_size=script_config.Animation.LocalWindowSize.Offset)
        append_to_log_file(logger, f"       ...Done")

        # Step 5: Collective plots
        append_to_log_file(logger, f"   Creating Collective Animation Frames...")

        collected_names = [
            getattr(attr, 'Name')
            for attr in (getattr(script_config.Animation, name) for name in dir(script_config.Animation))
            if getattr(attr, 'Collective', False)
        ]

        if collected_names:
            key = '-'.join(collected_names)
            tmp_dir = ensure_long_path_prefix(os.path.join(output_dir, "Animation-Frames", f"{key}-Plt-Files"))
            os.makedirs(tmp_dir, exist_ok=True)

            collected_arr = []
            for name in collected_names:
                collected_arr.append(data[name])

            filename = os.path.join(tmp_dir, f"{pltFile_name}.png")
            animation_frame_generation(
                data['X'],
                collected_arr,  # Note: val.Name must exist in pre_loaded_data
                collected_names,
                filename
            )
        append_to_log_file(logger, f"       ...Done")

        # Schlieren and Streamlines
        if script_config.Animation.Schlieren.Flag:
            append_to_log_file(logger, f"   Creating Schlieren Animation Frames...")
            tmp_dir = ensure_long_path_prefix(os.path.join(output_dir, "Animation-Frames", f"Schlieren-Plt-Files"))
            os.makedirs(tmp_dir, exist_ok=True)
            filename = os.path.join(tmp_dir, f"{pltFile_name}.png")
            # Generate Schlieren
            schlieren(dataset, filename)
            append_to_log_file(logger, f"       ...Done")

        if script_config.Animation.StreamLines.Flag:
            append_to_log_file(logger, f"   Creating StreamLines Animation Frames...")
            tmp_dir = ensure_long_path_prefix(
                os.path.join(output_dir, "Animation-Frames", f"StreamLines-Plt-Files"))
            os.makedirs(tmp_dir, exist_ok=True)
            filename = os.path.join(tmp_dir, f"{pltFile_name}.png")
            # Generate streamlines
            streamline(dataset, filename)
            append_to_log_file(logger, f"       ...Done")

        append_to_log_file(logger, f"\n")
        return result_dict
    else:
        return None


def pelec_processing(args, comm, logger):
    # Step 0: Collect current MPI rank and size
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Step 1: Unpack the arguments and initialize the output directory
    data_dirs, output_dir, script_config = args

    ###########################################
    # MPI Parallel Processing: Shared loading
    ###########################################
    # Step 2: Create an array to store the results
    result_arr = np.empty(len(data_dirs), dtype=object)

    if rank == 0:
        # Step 3: Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        append_to_log_file(logger, f"Output directory: {output_dir}")
        append_to_log_file(logger, f"Processing {len(data_dirs)} files with {size} MPI ranks.\n")

    # Step 4: Process each individual dataset
    for i, current_dir in enumerate(data_dirs):
        # Step 4.1: Load the data
        data = yt.load(current_dir)
        # Step 4.2: Process each file
        result_arr[i] = single_file_processing(data, (output_dir, script_config), comm, logger)

    # Ensure all ranks have completed processing
    comm.Barrier()

    ###########################################
    # Serial Processing: Only root does heavy post-processing
    ###########################################
    if rank == 0:
        append_to_log_file(logger, f"Global Result Variable Extraction...")

        print(result_arr[0], flush=True)

        # Step 5: Determine the wave velocities
        tmp_time = []
        tmp_flame_position = []
        tmp_shock_position = []
        for i in range(len(result_arr)):
            tmp_time.append(result_arr[i]['Time'])
            if script_config.Flame.Velocity.Flag or script_config.Flame.RelativeVelocity.Flag:
                tmp_flame_position.append(result_arr[i]['Flame']['Position'])
            if script_config.Shock.Velocity.Flag:
                tmp_shock_position.append(result_arr[i]['Shock']['Position'])

        if script_config.Flame.Velocity.Flag or script_config.Flame.RelativeVelocity.Flag:
            tmp_numerator = np.gradient(tmp_flame_position)
            tmp_denominator = np.gradient(tmp_time)
            tmp_flame_velocity = np.divide(tmp_numerator, tmp_denominator)

            append_to_log_file(logger, f"   Flame Velocity Extraction...")
            for i in range(len(result_arr)):
                if script_config.Flame.Velocity.Flag:
                    result_arr[i]['Flame']['Velocity'] = tmp_flame_velocity[i]
                if script_config.Flame.RelativeVelocity.Flag:
                    result_arr[i]['Flame']['Velocity'] = tmp_flame_velocity[i] - result_arr[i]['Flame']['Gas Velocity']
            append_to_log_file(logger, f"       ...Done")

        if script_config.Shock.Velocity.Flag:
            tmp_numerator = np.gradient(tmp_shock_position)
            tmp_denominator = np.gradient(tmp_time)

            append_to_log_file(logger, f"   Flame Releative Velocity Extraction...")
            for i in range(len(result_arr)):
                result_arr[i]['Shock']['Velocity'] = np.divide(tmp_numerator[i], tmp_denominator[i])
            append_to_log_file(logger, f"       ...Done")

        # Step 6: Save the results
        append_to_log_file(logger, f"Writing to File...")
        filename = os.path.join(output_dir, f"Processed-MPI-Global-Results-V-{version}.txt")
        write_output(result_arr, filename)
        append_to_log_file(logger, f"   ...Done")

        # Step 7: Generate the animations
        append_to_log_file(logger, f"Creating Animations...")
        parent_dir = ensure_long_path_prefix(os.path.join(output_dir, "Animation-Frames"))
        frame_dirs = [item for item in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, item))]
        for frame_dir in frame_dirs:
            filename = os.path.join(output_dir, f"{frame_dir}-Animation.mp4")
            generate_animation(os.path.join(parent_dir, frame_dir), filename)

        append_to_log_file(logger, f"   ...Done")

    return


########################################################################################################################
# Main Script
########################################################################################################################

def main(comm, logger):
    # Step 1: Initialize MPI
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Step 1: Define the script parameters
    data_parent_dir = '../2D-Test-Data'

    ddt_plt_dir = os.path.join(data_parent_dir, 'plt332330')
    # ddt_parent_dir = '../../../Domain-Length-284cm/0.09cm-Complete-Domain/Planar-Kernel-Level-6-Part-3'
    # ddt_plt_dir = os.path.join(data_parent_dir, 'ddt_plt')

    # Step 2: Set the processing parameters
    script_config = ScriptConfig()
    # Flame Parameters
    script_config.Flame.Position.Flag = True
    script_config.Flame.Velocity.Flag = True
    script_config.Flame.RelativeVelocity.Flag = True
    script_config.Flame.RelativeVelocity.Offset = 10e-6
    script_config.Flame.ThermodynamicState.Flag = True
    # script_config.Flame.ThermodynamicState.Offset = 10e-6
    script_config.Flame.HeatReleaseRate.Flag = True
    script_config.Flame.SurfaceLength.Flag = False
    script_config.Flame.FlameThickness.Flag = False
    script_config.Flame.ReynoldsNumber.Flag = False
    # Burned Gas Parameters
    script_config.BurnedGas.GasVelocity.Flag = True
    script_config.BurnedGas.ThermodynamicState.Flag = True
    # script_config.Flame.ThermodynamicState.Offset = 10e-6
    # Shock Parameters
    script_config.Shock.Position.Flag = True
    script_config.Shock.Velocity.Flag = True
    script_config.Shock.PreShockThermodynamicState.Flag = True
    script_config.Shock.PostShockThermodynamicState.Flag = True
    # Animation Parameters
    script_config.Animation.LocalWindowSize.Offset = 1e-3
    script_config.Animation.Temperature.Flag = True
    script_config.Animation.Temperature.Local = True
    script_config.Animation.Temperature.Collective = True
    script_config.Animation.Pressure.Flag = True
    script_config.Animation.Pressure.Local = True
    script_config.Animation.Pressure.Collective = True
    script_config.Animation.GasVelocity.Flag = True
    script_config.Animation.GasVelocity.Local = True
    script_config.Animation.HeatReleaseRate.Flag = True
    script_config.Animation.HeatReleaseRate.Local = True
    script_config.Animation.FlameGeometry.Flag = True
    script_config.Animation.FlameThickness.Flag = True
    script_config.Animation.Schlieren.Flag = False
    script_config.Animation.StreamLines.Flag = False

    # Step 3: Define the input parameters
    initialize_parameters(
        T=503.15,
        P=10.0 * 100000,
        Phi=1.0,
        Fuel='H2',
        mech='../Chemical-Mechanisms/Li-Dryer-H2-mechanism.yaml',
    )  # ✅ shared state
    # Add species if needed (Only use input_params.species if working with a small chemical mechanism file)
    add_species_vars(input_params.species)

    # Step 2: Determine the domain parameters
    row_idx = 0.0462731 + (8.7e-5 / 2)
    domain_info = domain_parameters(ddt_plt_dir, desired_y_location=row_idx)

    # Step 3: Log the domain information
    if rank == 0:
        append_to_log_file(logger, f"Domain Info Extracted at y = {domain_info[1][0][1]:.3g}m")

    script_config.DataExtraction.Location = domain_info[1][0][1]  # Set the location for data extraction
    script_config.DataExtraction.Grid = domain_info[-1]

    # Step 4: Create the result directories
    os.makedirs(os.path.join(f"Processed-MPI-Global-Results-V{version}", f"y-{domain_info[1][0][1]:.3g}cm"),
                exist_ok=True)
    output_dir = os.path.abspath(
        os.path.join(f"Processed-MPI-Global-Results-V{version}", f"y-{domain_info[1][0][1]:.3g}cm"))

    # Step 5: Process the data
    data_paths = load_directories(data_parent_dir)
    updated_data_paths = sort_files(data_paths)

    # Step 6: Process the data
    pelec_processing((updated_data_paths, output_dir, script_config), comm, logger)

    return

if __name__ == "__main__":
    # Step 1: Initialize the script and log file
    start_time = time.time()

    # Step 2: Initialize MPI and logger
    comm = MPI.COMM_WORLD
    print(f"Rank {comm.rank} of {comm.size} is running.", flush=True)

    if comm.rank == 0:
        logger = create_log_file("runlog")
        append_to_log_file(logger, f"Starting script with {comm.size} ranks...")
    else:
        logger = None
    logger = comm.bcast(logger, root=0)
    # Step 3: Run the main function
    main(comm, logger)

    # Step 4: Finalize the logger
    if comm.rank == 0:
        end_time = time.time()
        append_to_log_file(logger, f"Script completed successfully. Took {end_time - start_time:.2f} seconds.")
