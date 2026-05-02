# EMCC-Python: High-Performance Implementation of the Empirical Coupled-Channel Model

This repository contains a demo version of an optimized Python implementation of the Empirical Coupled-Channel (EMCC) model, designed for the calculation of capture cross sections. This program is provided only as a test version for performance optimization. Its implementation details differ from the EMCC model actually used in our paper, and therefore it does not guarantee a complete reproduction of the experimentally measured capture cross sections.
The primary contribution of this work is an improvement in computational efficiency. Through a combination of modern algorithmic restructuring and hardware-aware optimization, the execution time for complex potential surface scans has been reduced from approximately 7 hours in the original Fortran version to approximately 20 seconds in the current version.

## Methodological Improvements and Optimization

The performance gains were achieved through a multi-layered optimization strategy focusing on Just-In-Time (JIT) compilation, parallelization, and algorithmic refinement.

First, the core physics kernels—specifically the nuclear density integration and the folding potential calculations—were optimized using Numba. By utilizing the @njit decorator with fastmath enabled, these numerically intensive loops are compiled into optimized machine code via LLVM, effectively bypassing the Python interpreter overhead and achieving performance comparable to or exceeding compiled C/Fortran binaries. Vectorization techniques were further applied using NumPy broadcasting to handle batch calculations for distance arrays, minimizing explicit iteration.

Second, the scanning of deformation parameters, which represents the most time-consuming component of the simulation, was parallelized. The implementation utilizes the ProcessPoolExecutor to distribute the workload across all available CPU cores. This replaces the strictly serial execution model of the legacy code, allowing for simultaneous computation of potential surface points.

Third, the numerical algorithms for root-finding and integration were modernized. The inefficient manual step-loop methods used for locating the barrier radius and height were replaced with SciPy's implementation of Brent's method (brentq). This provides higher precision with significantly fewer function evaluations. Additionally, the numerical integration scheme was upgraded to a vectorized 3D Gauss-Legendre quadrature with precomputed nodes and weights, eliminating redundant runtime calculations.

## Input Configuration

The program requires an input file named "EMCCM.IN" structured as follows. Each line corresponds to specific physical parameters required by the model:

Line 1: Mass_P, Z_P, Mass_T, Z_T, Beta_P, Beta_T
        (Projectile and Target properties: Mass, Atomic Number, Deformation Parameter)
Line 2: E_min, E_max, E_step
        (Energy range and step size for the calculation in MeV)
Line 3: L_max, IVINT
        (Maximum angular momentum and Interaction switch)
Line 4: a_P, a_T
        (Diffuseness parameters for the nuclear density)

Example content for EMCCM.IN:
50, 22, 242, 94, 0.000, 0.237
189.68,  239.68,  1.
30, 2
0.55,0.55

## Usage and Dependencies

To execute this program, a Python environment (version 3.6) is required, along with the standard scientific computing stack: NumPy, SciPy, and Numba. These dependencies can be installed via standard package managers.

The program relies on an input configuration file named "EMCCM.IN", which must be present in the root directory. This file specifies the necessary physical parameters, including mass, charge, deformation parameters, and the energy range for the simulation. Upon execution of the main script ("EMCC_optimized.py"), the code initializes the potential shape, computes the barrier distribution function, and performs the cross-section calculations.

## Output Description

The simulation generates several data files containing the calculation results. "CROSS.DAT" provides the calculated capture cross sections as a function of center-of-mass energy. "TRANS.DAT" contains the transmission coefficients. Additionally, a barrier renormalization file ("barrier_factor.txt") is needed to adjust the barrier height and width if necessary.

## Copyright
This software is provided for academic and non-profit research purposes only. Commercial use, reproduction, or distribution of this software without prior written permission is strictly prohibited. If you use this code in your research, please acknowledge the author and cite the relevant publications.
