# view-factor-cuda

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A CUDA-based implementation of the MT View Factor calculation, featuring multi-GPU compuation using STL files as inputs.

The emitting, receiving, and blocking surfaces are provided to the solver in that order as input arguments. The following other settings are adjusted using environment variables:

- CUDA_VISIBLE_DEVICES: Adjust the available cards for use by the solver (comes from the CUDA toolkit).
- SELF_INT: Enables self-intersection using the following settings
    - 0 (default): no self intersection
    - 1: Self intersection on emitter
    - 2: Self intersection on receiver
    - 3: Self intersection on emitter and receiver

This work is prepared in collaboration with the Department of Mechanical Engineering at the University of Pittsburgh, and is meant to be a usable prototype for view factor calculation using the Müller Trumboure algorithm for self-intersection. The current work is in its final iteration, and all further work is going into an open-source framework for  

Authors:

- Katie Richmond -- Mechanical Engineering -- University of Pittsburgh
- Shane Riley -- Mechanical Engineering, Computer Science -- University of Pittsburgh
