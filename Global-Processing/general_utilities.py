import os
import re
import yt
import itertools
import multiprocessing

import cantera as ct

yt.set_log_level(0)

# Optional imports if needed externally:
__all__ = [
    'MyClass', 'input_params', 'initialize_parameters',
    'parallel_processing_function', 'init_pool',
    'ensure_long_path_prefix', 'sort_files', 'load_directories'
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
    iter_var, const_list, shared_input_params, predicate, kwargs = args
    global input_params
    input_params = shared_input_params
    return predicate((iter_var, const_list, kwargs))

def parallel_processing_function(iter_arr, const_list, predicate, n_procs=1, **kwargs):
    if n_procs > 1:
        with multiprocessing.Pool(
            processes=n_procs, initializer=init_pool, initargs=(input_params,)
        ) as pool:
            tasks = zip(
                iter_arr,
                itertools.repeat(const_list),
                itertools.repeat(input_params),
                itertools.repeat(predicate),
                itertools.repeat(kwargs)
            )
            results = pool.map(worker_function, tasks)
    else:
        results = [worker_function((val, const_list, input_params, predicate, kwargs)) for val in iter_arr]
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