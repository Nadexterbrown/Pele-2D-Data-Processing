import copy
import numpy as np
import pandas as pd
import cantera as ct

from scipy.linalg import eig

########################################################################################################################
# Function Scripts - Data Generation (Cantera)
########################################################################################################################

def cantera_enthalpy(data, input_params):

    # Map the temperature and pressure to arrays, and collect the species present in the solution
    temperature = data['Temperature']
    pressure = data['Pressure']
    species_list = [var for var in input_params.species]
    y_pts, x_pts, _ = data['Grid'].shape

    # Allocate memory for the array
    species_enthalpy = {
        f'Y({var})': np.full((y_pts, x_pts), np.nan)  # Initialize with NaNs
        for var in species_list
    }

    # Step 2: Create a temporary cantera gas object
    gas_tmp = ct.Solution(input_params.mech)

    for j in range(0, y_pts):
        for i in range(0, x_pts):
            if np.isnan(temperature[j, i]):
                continue  # Skip NaN values
            else:
                for k, species in enumerate(species_list):
                    # Map the species mass fractions to a dict for use with cantera gas object
                    species_dict = {
                        species: data[f"Y({species})"][j, i]  # Construct key dynamically
                        for species in species_list if f"Y({species})" in data  # Ensure key exists
                    }

                    # Set the gas object state
                    gas_tmp.TPY = temperature[j, i], pressure[j, i], species_dict

                    species_enthalpy[f"Y({species})"][j, i] = (gas_tmp.standard_enthalpies_RT[gas_tmp.species_index(species)] * ct.gas_constant * temperature[j,i]) / gas_tmp.molecular_weights[gas_tmp.species_index(species)]

    return species_enthalpy

########################################################################################################################
# Function Scripts - Mathematical Processes
########################################################################################################################
"""
def spatial_derivative(arr, dx, direction):
    ###########################################
    # Internal Functions
    ###########################################

    def derivative(dimensionality, i=None, j=None):

        if dimensionality == '1D':
            if 1 <= i < x_pts - 1 and not np.isnan(arr[i - 1]) and not np.isnan(arr[i + 1]):
                return (arr[i + 1] - arr[i - 1]) / (2 * dx)

                # Forward Difference (Second Order)
            elif i < x_pts - 2 and not np.isnan(arr[j, i + 1]) and not np.isnan(arr[j, i + 2]):
               return (-3 * arr[i] + 4 * arr[i + 1] - arr[i + 2]) / (2 * dx)

                # Backward Difference (Second Order)
            elif i > 1 and not np.isnan(arr[j, i - 1]) and not np.isnan(arr[j, i - 2]):
                return (3 * arr[i] - 4 * arr[i - 1] + arr[i - 2]) / (2 * dx)

        elif dimensionality == '2D-x':
            if 1 <= i < x_pts - 1 and not np.isnan(arr[j, i - 1]) and not np.isnan(arr[j, i + 1]):
                return (arr[j, i + 1] - arr[j, i - 1]) / (2 * dx)

            # Forward Difference (Second Order)
            elif i < x_pts - 2 and not np.isnan(arr[j, i + 1]) and not np.isnan(arr[j, i + 2]):
                return (-3 * arr[j, i] + 4 * arr[j, i + 1] - arr[j, i + 2]) / (2 * dx)

            # Backward Difference (Second Order)
            elif i > 1 and not np.isnan(arr[j, i - 1]) and not np.isnan(arr[j, i - 2]):
                return (3 * arr[j, i] - 4 * arr[j, i - 1] + arr[j, i - 2]) / (2 * dx)

        elif dimensionality == '2D-y':
            # Centered Difference (Second Order)
            if 1 <= j < y_pts - 1 and not np.isnan(arr[j - 1, i]) and not np.isnan(arr[j + 1, i]):
                return (arr[j + 1, i] - arr[j - 1, i]) / (2 * dx)

            # Forward Difference (Second Order)
            elif j < y_pts - 2 and not np.isnan(arr[j + 1, i]) and not np.isnan(arr[j + 2, i]):
                return (-3 * arr[j, i] + 4 * arr[j + 1, i] - arr[j + 2, i]) / (2 * dx)

            # Backward Difference (Second Order)
            elif j > 1 and not np.isnan(arr[j - 1, i]) and not np.isnan(arr[j - 2, i]):
                return (3 * arr[j, i] - 4 * arr[j - 1, i] + arr[j - 2, i]) / (2 * dx)

        elif dimensionality == '3D':
            raise ValueError("W.I.P. - Please choose another dimensionality")

        else:
            raise ValueError("Unsupported array dimensionality")

        return

    ###########################################
    # Main Function
    ###########################################

    # Step 1: Allocate memory to store the derivative
    dfdx = np.full_like(arr, np.nan)

    # Step 2: Extract the shape of arr
    pts_idx_arr = [(x, y) for x in range(arr.shape[1]) for y in range(arr.shape[0])]

    if arr.ndim == 1:
        x_pts = arr.shape[0]
        for x_idx in pts_idx_arr:
            dfdx[x_idx] = derivative('1D', i=x_idx)

    elif arr.ndim == 2:
        x_pts = arr.shape[1]
        y_pts = arr.shape[0]
        for x_idx, y_idx in pts_idx_arr:
            dfdx[y_idx, x_idx] = derivative(f'2D-{direction}', i=x_idx, j=y_idx)

    else:
        raise ValueError("Unsupported array dimensionality")

    return dfdx
"""

def spatial_derivative(f, x, direction):
    """
    Compute the spatial derivative using np.gradient with variable spacing.

    Parameters:
    arr : np.ndarray
        Input array (1D or 2D) containing function values.
    x : np.ndarray
        Grid spacing array, must match the corresponding axis of `arr`.
    direction : str
        Direction of differentiation ('x' or 'y' for 2D arrays).

    Returns:
    np.ndarray
        Array of the same shape as `arr` containing the computed derivative.
    """
    arr = np.asarray(f)  # Ensure input is a NumPy array
    x = np.asarray(x)

    if direction == 'x':
        dfdx = np.gradient(arr, axis=1, edge_order=2) / np.gradient(x, axis=1, edge_order=2)
    elif direction == 'y':
        try:
            dfdx = np.gradient(arr, axis=0, edge_order=2) / np.gradient(x, axis=0, edge_order=2)
        except:
            dfdx = np.zeros((1, len(arr.flatten())))
    else:
        raise ValueError("Direction must be 'x' or 'y' for 2D arrays.")

    return dfdx


########################################################################################################################
# Function Scripts - CEMA Processing
########################################################################################################################

def jacobian(data, input_params, species_enthalpy, jac_method='Simple', chem_method='Cantera'):
    ###########################################
    # Internal Functions
    ###########################################
    def cantera_solution(gas_tmp, T, P, Y):
        # Set the gas object state
        gas_tmp.TPY = T, P, Y

        # Calculate the species and temperature jacobian values
        species_jacobian = gas_tmp.net_production_rates_ddCi
        temperature_jacobian = gas_tmp.net_production_rates_ddT

        # Map the jacobians to a pandas dataframe
        species_jacobian = pd.DataFrame(species_jacobian, index=gas_tmp.species_names, columns=gas_tmp.species_names)
        temperature_jacobian = pd.Series(temperature_jacobian, index=gas_tmp.species_names)

        return species_jacobian, temperature_jacobian

    def pyjac_solution(gas_tmp, T, P, Y):
        # Set the gas object state
        gas_tmp.TPY = T, P, Y

        # Setup the state vector (Does not account for N2)
        y = np.zeros(gas_tmp.n_species)
        y[0] = T
        y[1:] = gas_tmp.Y[:-1]

        # Create a dydt vector
        dydt = np.zeros_like(y)
        pyjacob.py_dydt(0, P, y, dydt)

        # Create a jacobian vector
        jac = np.zeros(gas_tmp.n_species * gas_tmp.n_species)

        # Evaluate the Jacobian
        pyjacob.py_eval_jacobian(0, P, y, jac)

        J_vector = np.array(jac)  # Ensure it's a NumPy array
        J_tmp = J_vector.reshape((len(gas_tmp.n_species), len(gas_tmp.n_species)), order='F')  # Reshape using Fortran order
        J_matrix = np.zeros((J_tmp.shape[0] + 1, J_tmp.shape[0] + 1))
        J_matrix[:-1, -1] = J_matrix

        species_jacobian = J_matrix[1:, 1:]
        temperature_jacobian = J_matrix[1:, 0]

        # Map the jacobians to a pandas dataframe
        species_jacobian = pd.DataFrame(species_jacobian, index=gas_tmp.species_names, columns=gas_tmp.species_names)
        temperature_jacobian = pd.Series(temperature_jacobian, index=gas_tmp.species_names)

        return species_jacobian, temperature_jacobian

    def simple_jacobian(gas_tmp, T, P, Y):
        # Step 1: Allocate memory for the jacobian array
        tmp_jacobian = np.zeros((len(input_params.species), len(input_params.species)))

        # Set the gas object state
        gas_tmp.TPY = T, P, Y

        #
        if chem_method == 'Cantera':
            species_jacobian, temperature_jacobian = cantera_solution(gas_tmp, T, P, Y)
        elif chem_method == 'pyJac':
            if "pyjacob" not in globals():
                import pyjacob  # Only imports if not already loaded

            # reorder the gas to match pyJac
            n2_ind = gas_tmp.species_index('N2')
            specs = gas_tmp.species()[:]
            gas_tmp = ct.Solution(thermo='IdealGas', kinetics='GasKinetics',
                                  species=specs[:n2_ind] + specs[n2_ind + 1:] + [specs[n2_ind]],
                                  reactions=gas_tmp.reactions())

            species_jacobian, temperature_jacobian = pyjac_solution(gas_tmp, T, P, Y)
        else:
            print('Invalid Chemical Jacobian Solver Selection')

        # Calculate the CEMA jacobian values
        for row, primary_species in enumerate(input_params.species):
            for col, secondary_species in enumerate(input_params.species):

                if row < tmp_jacobian.shape[0] and col < tmp_jacobian.shape[1]:
                    tmp_jacobian[row, col] = species_jacobian[primary_species][secondary_species]

                elif row < tmp_jacobian.shape[0] and col == tmp_jacobian.shape[1]:
                    tmp_jacobian[row, col] = temperature_jacobian[primary_species]

                elif row == tmp_jacobian.shape[0] and col < tmp_jacobian.shape[1]:
                    tmp_jacobian[row, col] = (sum(species_jacobian[species][secondary_species] *
                                                  species_enthalpy[f'Y({species})'][j, i] for species in
                                                  input_params.species) /
                                              data['Cp'][j, i])

                elif row == tmp_jacobian.shape[0] and col == tmp_jacobian.shape[1]:
                    tmp_jacobian[row, col] = (sum(temperature_jacobian[species] *
                                                  gas_tmp.molecular_weights[gas_tmp.species_index(species)] *
                                                  species_enthalpy[f'Y({species})'][j, i] for species in
                                                  input_params.species) /
                                              data['Density'][j, i] / data['Cp'][j, i])

        return tmp_jacobian

    def compressible_jacobian(gas_tmp, T, P, Y):
        # Step 1: Allocate memory for the jacobian array
        if data['Dimension'] == 1:
            tmp_jacobian = np.zeros((len(input_params.species) + 2, len(input_params.species) + 2))
        else:
            tmp_jacobian = np.zeros((len(input_params.species) + 3, len(input_params.species) + 3))

        # Set the gas object state
        gas_tmp.TPY = T, P, Y

        #
        if chem_method == 'Cantera':
            species_jacobian, temperature_jacobian = cantera_solution(gas_tmp, T, P, Y)
        elif chem_method == 'pyJac':
            if "pyjacob" not in globals():
                import pyjacob  # Only imports if not already loaded

            # reorder the gas to match pyJac
            n2_ind = gas_tmp.species_index('N2')
            specs = gas_tmp.species()[:]
            gas_tmp = ct.Solution(thermo='IdealGas', kinetics='GasKinetics',
                                  species=specs[:n2_ind] + specs[n2_ind + 1:] + [specs[n2_ind]],
                                  reactions=gas_tmp.reactions())

            species_jacobian, temperature_jacobian = pyjac_solution(gas_tmp, T, P, Y)
        else:
            print('Invalid Chemical Jacobian Solver Selection')

        # Calculate the CEMA jacobian values
        for row in range(0, tmp_jacobian.shape[0]):
            if row < len(input_params.species):
                primary_species = input_params.species[row]

            for col in range(0, tmp_jacobian.shape[1]):
                if col < len(input_params.species):
                    secondary_species = input_params.species[col]

                if row < len(input_params.species) and col < len(input_params.species):
                    tmp_jacobian[row, col] = ((gas_tmp.molecular_weights[
                                                   gas_tmp.species_index(primary_species)] /
                                               gas_tmp.molecular_weights[
                                                   gas_tmp.species_index(secondary_species)]) *
                                              species_jacobian[primary_species][secondary_species] +
                                              (gas_tmp.molecular_weights[
                                                   gas_tmp.species_index(primary_species)] /
                                               data['Density'][j, i] * data['Cv'][j, i]) *
                                              (data['X Velocity'][j, i] ** 2 / 2 -
                                               (gas_tmp.standard_int_energies_RT[
                                                    gas_tmp.species_index(secondary_species)] *
                                                ct.gas_constant * data['Temperature'][j, i]) /
                                               gas_tmp.molecular_weights[
                                                   gas_tmp.species_index(secondary_species)])
                                              ) * temperature_jacobian[primary_species]

                elif row < len(input_params.species) and col == len(input_params.species):
                    tmp_jacobian[row, col] = (- (data['X Velocity'][j, i] *
                                                 gas_tmp.molecular_weights[
                                                     gas_tmp.species_index(primary_species)] /
                                                 data['Density'][j, i] / data['Cv'][j, i]) *
                                              temperature_jacobian[primary_species])

                elif row < len(input_params.species) and col == len(input_params.species) + 1 and data[
                    'Dimension'] == 2:
                    tmp_jacobian[row, col] = (- (data['Y Velocity'][j, i] *
                                                 gas_tmp.molecular_weights[
                                                     gas_tmp.species_index(primary_species)] /
                                                 data['Density'][j, i] / data['Cv'][j, i]) *
                                              temperature_jacobian[primary_species])

                elif row < len(input_params.species) and col == len(input_params.species) + data['Dimension']:
                    tmp_jacobian[row, col] = ((gas_tmp.molecular_weights[gas_tmp.species_index(primary_species)] /
                                               data['Density'][j, i] / data['Cv'][j, i]) *
                                              temperature_jacobian[primary_species])

                elif row >= len(input_params.species):
                    tmp_jacobian[row, col] = 0

                else:
                    print('Jacobian Error: Please Examine Code')

        return tmp_jacobian

    ###########################################
    # Main Function
    ###########################################
    # Step 1: Extract grid spacing from domain (constant grid spacing at set level)
    y_pts, x_pts, _ = data['Grid'].shape

    if jac_method == 'Simple':
        J = np.full((y_pts, x_pts, len(input_params.species), len(input_params.species)), np.nan)
    elif jac_method == 'Compressible':
        if data['Dimension'] == 1:
            J = np.full((y_pts, x_pts, len(input_params.species) + 2, len(input_params.species) + 2), np.nan)
        else:
            J = np.full((y_pts, x_pts, len(input_params.species) + 3, len(input_params.species) + 3), np.nan)

    #
    gas_tmp = ct.Solution(input_params.mech)
    #
    for j in range(0, y_pts):
        for i in range(0, x_pts):
            # Skip NaN values
            if np.isnan(data['Temperature'][j, i]):
                continue

            #
            if jac_method == 'Simple':
                tmp_jacobian = simple_jacobian(gas_tmp,
                                               data['Temperature'][j, i],
                                               data['Pressure'][j, i],
                                               {species: data[f'Y({species})'][j, i] for species in input_params.species})
            elif jac_method == 'Compressible':
                tmp_jacobian = compressible_jacobian(gas_tmp,
                                                     data['Temperature'][j, i],
                                                     data['Pressure'][j, i],
                                                     {species: data[f'Y({species})'][j, i] for species in input_params.species})

            #
            J[j, i] = tmp_jacobian

        return J

def cema_solver(data, input_params, solver_mode, chem_jac):
    ###########################################
    # Internal Functions
    ###########################################
    def simple_cema_terms():
        ###########################################
        # Main Function
        #
        # The process outlined here originates from discussion in Jaravel et al.
        # https://doi.org/10.1016/j.proci.2020.09.020
        #
        # Here we solve for the Jacobian of the modified energy equation:
        # Dgω(φ)/Dt = J ⋅ Dφ/Dt = J ⋅ gω(φ) + gf; J = ∂gω/∂φ
        ###########################################

        def diffusion_terms():
            ###########################################
            # Internal Functions
            ###########################################
            def directional_components(direction):
                ###########################################
                # Internal Functions
                ###########################################
                def initialize_variables():
                    # Species Mass Fraction Derivatives
                    dY = {
                        var: np.full((y_pts, x_pts), np.nan)  # Initialize with NaNs
                        for var in [f"Y({var})" for var in input_params.species]
                    }
                    # Diffusion Flux
                    J = {
                        var: np.full((y_pts, x_pts), np.nan)  # Initialize with NaNs
                        for var in [f"Y({var})" for var in input_params.species]
                    }
                    # Diffusion Flux Derivatives
                    dJ = {
                        var: np.full((y_pts, x_pts), np.nan)  # Initialize with NaNs
                        for var in [f"Y({var})" for var in input_params.species]
                    }
                    return dY, J, dJ

                ###########################################
                # Main Function
                ###########################################
                # Step 1: Extract the grid spacing from the domain
                if direction == 'x':
                    position = data['Grid'][:, :, 0]
                elif direction == 'y':
                    position = data['Grid'][:, :, 1]

                # Step 2: Initialize the variables
                dY, J, dJ = initialize_variables()

                # Step 3: Calculate the spatial derivatives
                for i, species in enumerate(input_params.species):
                    # Species Mass Fraction Derivatives
                    dY[f'Y({species})'] = spatial_derivative(data[f'Y({species})'], position, direction)
                    # Diffusion Flux
                    J[f'Y({species})'] = data['Density'] * data[f'D({species})'] * dY[f'Y({species})']
                    # Diffusion Flux Derivatives
                    dJ[f'Y({species})'] = spatial_derivative(J[f'Y({species})'], position, direction)

                # Temperature Spatial Derivative
                dT = spatial_derivative(data['Temperature'], position, direction)
                # Total Energy Flux (q_i)
                q = - data['Conductivity'] * dT + sum(J[f'Y({species})'] * species_enthalpy[f'Y({species})'] for species in input_params.species)
                # Total Energy Flux Derivative
                dq = spatial_derivative(q, position, direction)

                return dY, J, dJ, dT, q, dq

            ###########################################
            # Main Function
            ###########################################

            # Step 1:
            g_f = np.full((y_pts, x_pts, len(input_params.species)), np.nan)

            # Initialize and calculate the variables for the x-direction (and y-direction if applicable)
            dYdx, Jx, dJdx, dTdx, qx, dqdx = directional_components('x')

            if data['Dimension'] == 2:
                dYdy, Jy, dJdy, dTdy, qy, dqdy = directional_components('y')

            # Step 2: Calculate the full spatial array space
            for j in range(0, y_pts):
                for i in range(0, x_pts):
                    if np.isnan(data['Temperature'][j, i]):
                        continue  # Skip NaN values

                    for idx, species in enumerate(input_params.species):
                        g_f[j, i, idx] = - (dJdx[f'Y({species})'][j, i] + dJdy[f'Y({species})'][j, i]) / data['Density'][j, i]

                    g_f[j, i, -1] = - (dqdx[j, i] + dqdy[j, i]) / data['Density'][j, i] / data['Cp'][j, i]

            return g_f

        def chemical_terms():
            # Step 1:
            g_w = np.full((y_pts, x_pts, len(input_params.species)), np.nan)

            # Step 2: Calculate the full spatial array space
            for j in range(0, y_pts):
                for i in range(0, x_pts):
                    if np.isnan(data['Temperature'][j, i]):
                        continue  # Skip NaN values

                    for idx, species in enumerate(input_params.species):
                        g_w[j, i, idx] = data[f'W({species})'][j, i] / data['Density'][j, i]

                    g_w[j, i, -1] = - (sum(data[f'W({species})'][j, i] * species_enthalpy[f'Y({species})'][j, i] for species in input_params.species)) / data['Density'][j, i] / data['Cp'][j, i]

            return g_w

        ###########################################
        # Main Function
        ###########################################
        # Step 1: Calculate the diffusion terms for CEMA analysis
        g_f = diffusion_terms()

        # Step 2: Calculate the reaction terms for CEMA analysis
        g_w = chemical_terms()

        return g_f, g_w

    def compressible_cema_terms():
        ###########################################
        # Main Function
        #
        # The process outlined here originates from discussion in Ren et al.
        # https://doi.org/10.2514/1.J057994
        #
        # Here we solve for the Jacobian of the modified energy equation:
        # Dgω(φ)/Dt = J ⋅ Dφ/Dt = J ⋅ gω(φ) + gf; J = ∂gω/∂φ
        ###########################################

        def diffusion_terms():
            ###########################################
            # Internal Functions
            ###########################################
            def directional_components(direction):
                ###########################################
                # Internal Functions
                ###########################################
                def initialize_variables():
                    # Species Mass Fraction Derivatives
                    dY = {
                        var: np.full((y_pts, x_pts), np.nan)  # Initialize with NaNs
                        for var in [f"Y({var})" for var in input_params.species]
                    }
                    # Diffusion Flux
                    J = {
                        var: np.full((y_pts, x_pts), np.nan)  # Initialize with NaNs
                        for var in [f"Y({var})" for var in input_params.species]
                    }
                    # Diffusion Flux Derivatives
                    dJ = {
                        var: np.full((y_pts, x_pts), np.nan)  # Initialize with NaNs
                        for var in [f"Y({var})" for var in input_params.species]
                    }
                    return dY, J, dJ

                ###########################################
                # Main Function
                ###########################################
                # Step 1: Extract the grid spacing from the domain
                if direction == 'x':
                    position = data['Grid'][:, :, 0]
                elif direction == 'y':
                    position = data['Grid'][:, :, 1]

                # Step 2: Initialize the variables
                dY, J, dJ = initialize_variables()

                # Step 3: Calculate the spatial derivatives
                for i, species in enumerate(input_params.species):
                    # Species Mass Fraction Derivatives
                    dY[f'Y({species})'] = spatial_derivative(data[f'Y({species})'], position, direction)
                    # Diffusion Flux
                    J[f'Y({species})'] = data['Density'] * data[f'D({species})'] * dY[f'Y({species})']
                    # Diffusion Flux Derivatives
                    dJ[f'Y({species})'] = spatial_derivative(J[f'Y({species})'], position, direction)

                # Temperature Spatial Derivative
                dT = spatial_derivative(data['Temperature'], position, direction)

                # Pressure Spatial Derivative
                dP = spatial_derivative(data['Pressure'], position, direction)

                # Velocity Spatial Derivatives
                du = spatial_derivative(data['X Velocity'], position, direction)
                dv = spatial_derivative(data['Y Velocity'], position, direction)

                # Pressure Velocity Derivatives
                dPu = spatial_derivative(data['Pressure'] * data['X Velocity'], position, direction)
                dPv = spatial_derivative(data['Pressure'] * data['Y Velocity'], position, direction)

                return dY, J, dJ, dT, dP, du, dv, dPu, dPv

            ###########################################
            # Main Function
            ###########################################
            # Following the approach from Ren et al., here we calculate the components of the diffusion source terms
            # (g_d, g_r, g_T)

            # Step 1:
            if data['Dimension'] == 1:
                g_d = np.full((y_pts, x_pts, len(input_params.species) + 2), np.nan)
                g_r = np.full((y_pts, x_pts, len(input_params.species) + 2), np.nan)
                g_T = np.full((y_pts, x_pts, len(input_params.species) + 2), np.nan)
            else:
                g_d = np.full((y_pts, x_pts, len(input_params.species) + 3), np.nan)
                g_r = np.full((y_pts, x_pts, len(input_params.species) + 3), np.nan)
                g_T = np.full((y_pts, x_pts, len(input_params.species) + 3), np.nan)

            # Initialize and calculate the variables for the x-direction (and y-direction if applicable)
            dYdx, Jx, dJdx, dTdx, dPdx, dudx, dvdx, dPudx, dPvdx = directional_components('x')
            dYdy, Jy, dJdy, dTdy, dPdy, dudy, dvdy, dPudy, dPvdy = directional_components('y')
            species_enthalpy = cantera_enthalpy(data, input_params)

            tau_xx = data['Viscosity'] * (dudx + dudx)
            tau_xy = data['Viscosity'] * ((dudy + dvdx) - (2 / 3) * (dudx + dvdy))
            tau_yx = data['Viscosity'] * ((dvdx + dudy) - (2 / 3) * (dudx + dvdy))
            tau_yy = data['Viscosity'] * (dvdy + dvdy)

            dtau_1 = spatial_derivative(tau_xx, data['Grid'][:, :, 0], 'x') + spatial_derivative(tau_xy,
                                                                                                 data['Grid'][:, :, 1],
                                                                                                 'y')
            dtau_2 = spatial_derivative(tau_yx, data['Grid'][:, :, 0], 'x') + spatial_derivative(tau_yy,
                                                                                                 data['Grid'][:, :, 1],
                                                                                                 'y')

            g_d_energ_1 = (data['Conductivity'] * dTdx +
                           sum(data['Density'] * data[f'D({species})'] * species_enthalpy[f'Y({species})'] * dYdx[
                               f'Y({species})'] for species in input_params.species) +
                           data['X Velocity'] * tau_xx + data['Y Velocity'] * tau_yx)

            g_d_energ_2 = (data['Conductivity'] * dTdy +
                           sum(data['Density'] * data[f'D({species})'] * species_enthalpy[f'Y({species})'] * dYdy[
                               f'Y({species})'] for species in input_params.species) +
                           data['X Velocity'] * tau_xy + data['Y Velocity'] * tau_yy)

            dg_d_energ_1_dx = spatial_derivative(g_d_energ_1, data['Grid'][:, :, 0], 'x')
            dg_d_energ_2_dy = spatial_derivative(g_d_energ_2, data['Grid'][:, :, 1], 'y')

            # Step 5: Calculate the full spatial array space
            for j in range(0, y_pts):
                for i in range(0, x_pts):
                    # Calculate the species terms
                    for idx, species in enumerate(input_params.species):
                        g_d[j, i, idx] = dJdx[f'Y({species})'][j, i] + dJdy[f'Y({species})'][j, i]
                        g_r[j, i, idx] = - data['Density'][j, i] * data[f'Y({species})'][j, i] * (
                                    dudx[j, i] + dvdy[j, i])
                        if not np.isnan(g_r[j, i, idx]):
                            g_T[j, i, idx] = 0

                    # Calculate the advection terms
                    g_d[j, i, -3] = dtau_1[j, i]
                    g_d[j, i, -2] = dtau_2[j, i]

                    g_r[j, i, -3] = -data['Density'][j, i] * data['X Velocity'][j, i] * (dudx[j, i] + dvdy[j, i])
                    g_r[j, i, -2] = -data['Density'][j, i] * data['Y Velocity'][j, i] * (dudx[j, i] + dvdy[j, i])

                    g_T[j, i, -3] = -dPdx[j, i]
                    g_T[j, i, -2] = -dPdy[j, i]

                    # Calculate the energy terms
                    g_d[j, i, -1] = dg_d_energ_1_dx[j, i] + dg_d_energ_2_dy[j, i]

                    g_r[j, i, -1] = -data['rho_e'][j, i] * (dudx[j, i] + dvdy[j, i])

                    g_T[j, i, -1] = -(dPudx[j, i] + dPvdy[j, i])

            return g_d, g_r, g_T

        def chemical_terms():
            # Step 1: Pre-allocate space for the arrays
            if data['Dimension'] == 1:
                g_w = np.full((y_pts, x_pts, len(input_params.species) + 2), np.nan)
            else:
                g_w = np.full((y_pts, x_pts, len(input_params.species) + 3), np.nan)

            # Step 3: Calculate the full spatial array space
            for j in range(0, y_pts):
                for i in range(0, x_pts):
                    # Calculate the species terms
                    for idx, species in enumerate(input_params.species):
                        g_w[j, i, idx] = data[f'W({species})'][j, i]

                    if not np.isnan(g_w[j, i, 0]):
                        if data['Dimension'] == 1:
                            # Calculate the advection terms
                            g_w[j, i, -2] = 0
                            # Calculate the energy terms
                            g_w[j, i, -1] = 0
                        else:
                            # Calculate the advection terms
                            g_w[j, i, -3] = 0
                            g_w[j, i, -2] = 0

                            # Calculate the energy terms
                            g_w[j, i, -1] = 0  # Skip NaN values

            return g_w

        ###########################################
        # Main Function
        ###########################################

        # Step 1: Extract grid spacing from domain (constant grid spacing at set level)
        y_pts, x_pts, _ = data['Grid'].shape

        # Step 2: Calculate the diffusion terms for CEMA analysis
        g_d, g_r, g_T = diffusion_terms()

        # Step 3: Calculate the reaction terms for CEMA analysis
        g_w = chemical_terms()

        return g_w, g_d, g_r, g_T

    ###########################################
    # Main Function
    ###########################################

    # Step 1: Extract grid spacing from domain (constant grid spacing at set level)
    y_pts, x_pts, _ = data['Grid'].shape

    # Step 2: Determine the species enthalpy
    species_enthalpy = cantera_enthalpy(data, input_params)

    # Step 3: Calculate the Jacobian
    print("     Calculating Jacobian...")
    J = jacobian(data, input_params, species_enthalpy, jac_method=solver_mode, chem_method=chem_jac)

    # Step 4: Determine the eigenstate at each point
    print("     Calculating Eigenvalues...")
    if solver_mode == 'Simple':
        eig_val = np.full((y_pts, x_pts, len(input_params.species)), np.nan)
        eig_vec = np.full((y_pts, x_pts, len(input_params.species), len(input_params.species)), np.nan)
    elif solver_mode == 'Compressible':
        if data['Dimension'] == 1:
            eig_val = np.full((y_pts, x_pts, len(input_params.species) + 2), np.nan)
            eig_vec = np.full((y_pts, x_pts, len(input_params.species) + 2, len(input_params.species) + 2), np.nan)
        else:
            eig_val = np.full((y_pts, x_pts, len(input_params.species) + 3), np.nan)
            eig_vec = np.full((y_pts, x_pts, len(input_params.species) + 3, len(input_params.species) + 3), np.nan)

    for j in range(0, y_pts):
        for i in range(0, x_pts):
            J_local = J[j, i]
            if np.isnan(J_local).any():
                continue  # Skip NaN values

            eig_val[j, i], eig_vec[j, i], _ = eig(J_local, left=True)

    # Step 3: Calculate the diffusion and chemical terms for CEMA analysis
    print("     Calculating CEMA Terms...")
    if solver_mode == 'Simple':
        g_f, g_w = simple_cema_terms()

    elif solver_mode == 'Compressible':
        g_w, g_d, g_r, g_T = compressible_cema_terms()

    lambda_e = np.zeros((y_pts, x_pts))
    phi_w = np.full((y_pts, x_pts), np.nan)
    phi_f = np.full((y_pts, x_pts), np.nan)
    phi_d = np.full((y_pts, x_pts), np.nan)
    phi_r = np.full((y_pts, x_pts), np.nan)
    phi_T = np.full((y_pts, x_pts), np.nan)
    alpha = np.full((y_pts, x_pts), np.nan)
    for j in range(y_pts):
        for i in range(x_pts):
            # Step 2: Determine the mode with the largest positive real part eigenvalue
            if np.isnan(eig_val[j, i, :]).any():
                continue  # Skip NaN values

            local_val = eig_val[j, i, :].real
            local_vec = eig_vec[j, i, :, :].real

            valid_mask = local_val > 0
            if local_val[valid_mask].size == 0:
                continue

            eig_idx = np.argmax(local_val[valid_mask])
            lambda_e[j, i] = local_val[valid_mask][eig_idx].real
            b_e = (local_vec[:, valid_mask].real)[:, eig_idx]
            """
            # Treat all eigenvalues as an explosive mode (i.e., the mode with the largest real part)
            if (local_val).size == 0:
                continue

            eig_idx = np.argmax(local_val)
            lambda_e[j, i] = local_val[eig_idx].real
            b_e = local_vec[:, eig_idx].real
            """
            # Compute phi_w before applying the sign correction
            phi_w[j, i] = np.dot(b_e, g_w[j, i])

            # Ensure phi_w is non-negative by flipping b_e if needed
            if phi_w[j, i] < 0:
                b_e *= -1
                phi_w[j, i] = -phi_w[j, i]  # Also correct phi_w

            if solver_mode == 'Simple':
                phi_f[j, i] = np.dot(b_e, g_f[j, i])

            elif solver_mode == 'Compressible':
                phi_d[j, i] = np.dot(b_e, g_d[j, i])
                phi_r[j, i] = np.dot(b_e, g_r[j, i])
                phi_T[j, i] = np.dot(b_e, g_T[j, i])
                phi_f[j, i] = np.dot(b_e, g_d[j, i] + g_r[j, i] + g_T[j, i])

            alpha[j, i] = abs(phi_f[j, i] / phi_w[j, i]) if phi_w[j, i] != 0 else np.nan

    if solver_mode == 'Simple':
        return phi_w, phi_f, alpha
    elif solver_mode == 'Compressible':
        return phi_w, phi_d, phi_r, phi_T, phi_f, alpha
