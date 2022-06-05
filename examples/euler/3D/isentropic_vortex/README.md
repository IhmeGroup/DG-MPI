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
2. Add the `set_state_*` function for the initial condition to the corresponding [`functions.h`](https://github.com/IhmeGroup/DG-MPI/blob/1ba02e31faf8be06967b9f7b3fc03d46d998075d/src/physics/euler/functions.h#L36) and [`functions.cpp`](https://github.com/IhmeGroup/DG-MPI/blob/1ba02e31faf8be06967b9f7b3fc03d46d998075d/src/physics/euler/functions.cpp#L147) (This is already completed for this case so it does not need to be recompiled). Note here that the function for the 3D isentropic vortex case has a definition for both 2D and 3D. Here, we are only discussing the 3D version. This function takes in the physics object, the physical coordinates in space, and the simulation time and returns the state at the quadrature points. Any initial condition function would have a very similar format. We note also that if an analytical solution exists, it is provided as the initial condition, not as a separate function.
3. Add the call to the IC in [`physics/base.cpp`](https://github.com/IhmeGroup/DG-MPI/blob/1e98593cb924104a2037c1a3b971d56682a3fa2b/src/physics/base/base.cpp#L90). This is done via if statements that compare the initial condition provided in the `input.toml` file versus the list of enums in `enums.h`. Note: This is again templated on dimension and there is a 2D and 3D version.
4. Add the call to the exact solution to [`physics/base.cpp`](https://github.com/IhmeGroup/DG-MPI/blob/1e98593cb924104a2037c1a3b971d56682a3fa2b/src/physics/base/base.cpp#L123) if the initial condition can also be the analytical solution to the case. 

## Step 3: Take a look at the `input.toml` file.

This is a fairly simple file that provides the basis inputs to the solver. Users can change the location of the mesh file (here, we default to the 4^3 case), the type of basis function, the time-stepping scheme, the simulation order, and more. For the first run, this file does not need to be modified unless you would like to change the mesh that is being used.

## Step 4: Submit the applicable slurm script.

There are two job submission scripts (`runslurm_cpu.sh` and `runslurm_gpu.sh`). Each is setup to either run the case using the CPU or GPU solver respectively. The scripts use relative paths so they should correctly point to the executable file for the appropriate solver. Note: These should be used on Stanford's Yellowstone cluster, but they could be easily modified to run for any other machine if needed. Submit using the following command:
   ```
   sbatch runslurm_cpu.sh
   ```
   
## Step 5: Look at the log file and make sure things ran as expected.

On yellowstone, a log file will automatically be generated that includes the outputs from the code. At the end of the simulation, a file called `data.h5` will be generated that has information for post-processing.

## Step 6: Calculate the L2-norm error.

To calculate the L2-norm error for the simulation use `srun post.sh`. This will read in the `data.h5` file and use the analytical solution (given by the initial condition) to calculate the error in the simulation. If you run with the default settings you can then compare what is output to your command line against the numbers in [`regression_data.txt`](https://github.com/IhmeGroup/DG-MPI/blob/main/examples/euler/3D/isentropic_vortex/regression_data.txt). Each entry corresponds to the L2-norm of `rho`, `rhou`, `rhov`, `rhow`, `rhoE` respectivally.

## Step 7: Try different things out!

Now try playing with the different settings. Change to the LegendreHex basis, do you get the same answer? Try a different time stepping scheme. Try a finer mesh. If the mesh is too fine try increasing the node count (Note: currently only supported for CPUs).
