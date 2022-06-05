# 3D Isentropic Vortex Tutorial

This is the primary test case for the DG-MPI solver. It can be used to verify the implementation of the DG discritization for the Euler equations as well as be used for testing the multi-node scaling via MPI.

## Step 1: Download the meshes

The meshes for this case are stored on google drive. To automatically download the meshes use the following commands (assuming you are in the `isentropic_vortex` directory:

   ```
   cd meshes
   python download_mesh_pkg.py
   ```
Note: you will need to have `gdown` installed as a python library. I used `conda install gdown`, but I believe it can also be installed via `pip`.

## Step 2: Take a look at the initial condition / exact solution source code

The way to add a new initial condition uses the following procedure:
1. Add the name of the initial condition to [`enums.h`](https://github.com/IhmeGroup/DG-MPI/blob/1ba02e31faf8be06967b9f7b3fc03d46d998075d/src/common/enums.h#L93).
2. Add the `set_state_*` function for the initial condition to the corresponding [`functions.h`](https://github.com/IhmeGroup/DG-MPI/blob/1ba02e31faf8be06967b9f7b3fc03d46d998075d/src/physics/euler/functions.h#L36) and [`functions.cpp`](https://github.com/IhmeGroup/DG-MPI/blob/1ba02e31faf8be06967b9f7b3fc03d46d998075d/src/physics/euler/functions.cpp#L147). Note here that the function for the 3D isentropic vortex case has a definition for both 2D and 3D. Here, we are only discussing the 3D version. This function takes in the physics object, the physical coordinates in space, and the simulation time and returns the state at the quadrature points.
