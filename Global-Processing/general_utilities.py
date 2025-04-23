import os
import re
import yt
import itertools
import multiprocessing
import cantera as ct
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.animation as animation

yt.set_log_level(0)

# Optional imports if needed externally:
__all__ = [
    'MyClass', 'input_params', 'initialize_parameters',
    'parallel_processing_function', 'init_pool',
    'ensure_long_path_prefix', 'sort_files', 'load_directories', 'write_output',
    'animation_frame_generation', 'generate_animation',
]


#################################################################
# Initial Thermodynamic Condition Class
#################################################################
class MyClass:
    def __init__(self):
        self.T = None
        self.P = None
        self.Phi = None
        self.Fuel = None
        self.mech = None
        self.species = None
        self.oxygenAmount = None
        self.nitrogenAmount = None
        self.X = {}

    def update_composition(self):
        if self.Fuel == "H2":
            self.oxygenAmount = 0.5
        elif self.Fuel == "C2H6":
            self.oxygenAmount = 3.5
        elif self.Fuel == "C4H10":
            self.oxygenAmount = 6.5
        else:
            raise ValueError(f"Unknown fuel type: {self.Fuel}")

        self.X = {
            self.Fuel: self.Phi,
            'O2': self.oxygenAmount,
            'N2': self.nitrogenAmount
        }

    def load_mechanism_species(self):
        if not self.mech:
            raise ValueError("Mechanism file not specified.")
        try:
            gas = ct.Solution(self.mech)
            self.species = gas.species_names
            del gas
        except Exception as e:
            raise RuntimeError(f"Failed to load mechanism file: {e}")


input_params = MyClass()


def initialize_parameters(T, P, Phi, Fuel, mech, nitrogenAmount=0):
    input_params.T = T
    input_params.P = P
    input_params.Phi = Phi
    input_params.Fuel = Fuel
    input_params.mech = mech
    input_params.nitrogenAmount = nitrogenAmount
    input_params.update_composition()
    input_params.load_mechanism_species()


#################################################################
# Multiprocessing and Utility Functions
#################################################################
def worker_function(args):
    iter_var, const_list, predicate, kwargs = args
    # Since input_params is already global, no need to pass it as an argument
    return predicate((iter_var, const_list, kwargs))


def parallel_processing_function(iter_arr, const_list, predicate, n_procs=1, **kwargs):
    if n_procs > 1:
        with multiprocessing.Pool(
                processes=n_procs, initializer=init_pool, initargs=(input_params,)  # Pass global input_params here
        ) as pool:
            tasks = zip(
                iter_arr,
                itertools.repeat(const_list),
                itertools.repeat(predicate),
                itertools.repeat(kwargs)
            )
            results = pool.map(worker_function, tasks)
    else:
        # In serial mode, simply pass input_params
        results = [worker_function((val, const_list, predicate, kwargs)) for val in iter_arr]

    return results


def init_pool(global_params):
    global input_params
    input_params = global_params


def ensure_long_path_prefix(path):
    if path.startswith(r"\\"):
        return r"\\?\UNC" + path[1:]
    return r"\\?\\" + path


def sort_files(file_list):
    def extract_number(file_path):
        folder = os.path.basename(file_path)
        match = re.search(r'plt(\d+)', folder)
        return int(match.group(1)) if match else float('inf')

    return sorted(file_list, key=extract_number)


#################################################################
# Directory and File Management
#################################################################
def load_directories(relative_dir):
    if not os.path.exists(os.path.join(os.path.dirname(relative_dir))):
        raise FileNotFoundError(f"Directory does not exist: {relative_dir}")

    # Step 2: Collect all the present pelec data directories
    dir_path = [
        os.path.join(relative_dir, name)
        for name in os.listdir(relative_dir)
        if os.path.isdir(os.path.join(relative_dir, name)) and name.startswith('plt')
    ]

    return dir_path


def write_output(data_dict, output_dir):

    ###########################################
    # Internal Functions
    ###########################################

    UNIT_MAP = {
        'Position': 'cm',
        'Temperature': 'K',
        'Pressure': 'kg / m / s^2',
        'Density': 'kg / m^3',
        'Viscosity': 'kg / m / s',
        'Conductivity': 'kg m^2 / s^3 / m / K',
        'Sound speed': 'm / s',
        'Mach Number': '',
        'Velocity': 'm / s',
        'Gas Velocity': 'm / s',
        'Heat Release Rate': 'kg m^2 / s^3 / m^3',
        'Cp': 'g m^2 / s^2 / kg / K',
        'Cv': 'g m^2 / s^2 / kg / K',
    }

    FIELD_WIDTH = 55

    def collect_headers_by_group(nested_dict):
        grouped_headers = {}

        for group, sub_dict in nested_dict.items():
            if isinstance(sub_dict, dict):
                headers = []

                def recurse(sub, prefix=""):
                    for key, value in sub.items():
                        if isinstance(value, dict):
                            recurse(value, prefix + key + " ")
                        else:
                            unit = UNIT_MAP.get(key.strip(), "")
                            unit_str = f" [{unit}]" if unit else ""
                            headers.append(prefix + key + unit_str)

                recurse(sub_dict)
                grouped_headers[group] = [f"{group} {h}" for h in headers]
            else:
                grouped_headers[group] = [f"{group}"]

        return grouped_headers

    def collect_flat_data(nested_dict):
        data_rows = []

        num_rows = len(next(iter(next(iter(nested_dict.values())).values())))  # assumes consistent length

        for i in range(num_rows):
            row = []

            def recurse(sub):
                for value in sub.values():
                    if isinstance(value, dict):
                        recurse(value)
                    else:
                        val = value[i]
                        if isinstance(val, (list, tuple, np.ndarray)):
                            row.extend(val)
                        else:
                            row.append(val)

            for group in nested_dict.values():
                if isinstance(group, dict):
                    recurse(group)
                else:
                    row.append(group[i])

            data_rows.append(row)

        return data_rows

    def write_nested_dict_to_file(nested_dict, filename):
        headers_grouped = collect_headers_by_group(nested_dict)
        headers_flat = [h for group in headers_grouped.values() for h in group]
        data_rows = collect_flat_data(nested_dict)

        with open(filename, 'w') as f:
            # Index row
            index_line = "# " + "".join(f"{i + 1:<{FIELD_WIDTH}d}" for i in range(len(headers_flat)))
            f.write(index_line + "\n")

            # Header row
            header_line = "# " + "".join(f"{h:<{FIELD_WIDTH}s}" for h in headers_flat)
            f.write(header_line + "\n")

            # Data rows
            for row in data_rows:
                data_line = "".join(f"{float(v):<{FIELD_WIDTH}.6e}" if isinstance(v, (
                int, float, np.floating)) else f"{str(v):<{FIELD_WIDTH}}" for v in row)
                f.write(data_line + "\n")

    ###########################################
    # Main Function
    ###########################################

    write_nested_dict_to_file(data_dict, output_dir)

    return


#################################################################
# Animation and Visualization
#################################################################
def animation_frame_generation(x, y, labels, output_dir, plot_center=None, window_size=None):
    """
       x_list: list of arrays for x-axis data
       y_list: list of arrays for y-axis data
       labels: optional list of labels for each y series
    """

    ###########################################
    # Internal Functions
    ###########################################

    def plot_axis(x, y):
        if plot_center is not None:
            indices = np.where(
                (x >= plot_center - window_size / 2) &
                (x <= plot_center + window_size / 2)
            )[0]
            return x[indices], y[indices]
        return x, y

    def plot_figure():
        fig, ax = plt.subplots()
        axes = [ax]

        # First axis
        x0, y0 = plot_axis(x, y[0])
        ax.plot(x0, y0, label=labels[0], color=f"C0")
        ax.set_ylabel(labels[0])
        ax.set_xlabel("x")
        ax.tick_params(axis='y', labelcolor=f"C0")

        # Additional y-axes, spaced to avoid overlap
        for i in range(1, len(y)):
            ax_new = ax.twinx()
            axes.append(ax_new)

            # Shift spine to the right
            offset = 0.1 * (i - 1)
            ax_new.spines["right"].set_position(("axes", 1 + offset))
            ax_new.set_frame_on(True)  # ensure visibility
            ax_new.patch.set_visible(False)  # don't obscure background

            # Plot data
            _, yi = plot_axis(x, y[i])
            ax_new.plot(x0, yi, label=labels[i], color=f"C{i}")
            ax_new.set_ylabel(labels[i])
            ax_new.tick_params(axis='y', labelcolor=f"C{i}")

        fig.tight_layout()
        fig.legend(loc="upper right")
        return fig

    ###########################################
    # Main Function
    ###########################################
    if not isinstance(y, list):
        y = [y]
    if labels is None or isinstance(labels, str):
        labels = [labels]

    assert all(len(x) == len(yi) for yi in y), "All y arrays must match length of x"
    if labels is None:
        labels = [f"Series {i}" for i in range(len(y))]

    # Save plot
    fig = plot_figure()
    plt.tight_layout()
    fig.savefig(output_dir, format='png')
    plt.close(fig)


def generate_animation(source_dir, output_dir):
    # Extract numeric parts of filenames for sorting
    def extract_frame_number(filename):
        match = re.search(r'plt(\d+)', filename)
        return int(match.group(1)) if match else -1

    # Filter and sort files by frame number
    image_files = [
        os.path.join(source_dir, f)
        for f in sorted(os.listdir(source_dir), key=extract_frame_number)
        if f.endswith('.png')  # Only include PNG files
    ]

    if not image_files:
        print("No image files found in the folder.")
        return

    # Load the first image to determine figure size
    first_image = mpimg.imread(image_files[0])
    fig = plt.figure(figsize=(first_image.shape[1] / 100, first_image.shape[0] / 100), dpi=100)
    plt.axis('off')
    img_display = plt.imshow(first_image)

    # Animation update function
    def update(frame):
        img_display.set_array(mpimg.imread(image_files[frame]))
        return img_display,

    # Create and save the animation
    ani = animation.FuncAnimation(fig, update, frames=len(image_files), blit=True)
    writer = animation.FFMpegWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    ani.save(output_dir, writer=writer)
    plt.close(fig)
