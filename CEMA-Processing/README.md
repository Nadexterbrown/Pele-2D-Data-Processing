# CEMA-Processing

## Overview

This repository contains scripts for processing and analyzing data from Direct Numerical Simulations (DNS) using the PeleC solver. The primary focus is on Chemical Explosive Mode Analysis (CEMA) to study flame dynamics and detonation phenomena.

## Features

- **Data Import and Export**: Functions to read and write simulation data.
- **Parallel Processing**: Utilizes multiprocessing for efficient data handling.
- **Flame Contour Extraction**: Methods to determine flame contours and normal vectors.
- **CEMA Source Terms Calculation**: Computes source terms for CEMA.
- **Jacobian and Mode Analysis**: Determines the Jacobian matrix and analyzes explosive modes.

## Requirements

- Python 3.x
- Required Python packages:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `scipy`
  - `yt`
  - `multiprocessing`
  - `itertools`
  - `re`
  - `cantera`
  - `geomdl`

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/CEMA-Processing.git
   ```
2. Install the required packages:
   ```sh
   pip install -r requirements.txt
   ```

## Usage

1. **Initialize Parameters**: Set up the initial parameters for the simulation.
   ```python
   initialize_parameters(
       T=503.15,
       P=10.0 * 100000,
       Phi=1.0,
       Fuel='H2',
       mech='Li-Dryer-H2-mechanism.yaml',
   )
   ```

2. **Data Import**: Load the simulation data.
   ```python
   data = data_import('path/to/datafile.txt')
   ```

3. **Down Sampling**: Apply down sampling to the data.
   ```python
   data_ds = down_sampling(data, down_sampling_dt=1/160000)
   ```

4. **CEMA Analysis**: Perform CEMA analysis on the data.
   ```python
   g_w, g_d, g_r, g_T = cema_source_terms(data)
   flame_pts, jacobian_arr = cema_jacobian(data, g_w)
   phi_w, phi_f, alpha = cema_mode(data, jacobian_arr, g_w, g_d, g_r, g_T, flame_pts)
   ```

## Notes

- This project is a work in progress. Some features and functions may not be fully implemented or tested.
- Contributions and feedback are welcome.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
