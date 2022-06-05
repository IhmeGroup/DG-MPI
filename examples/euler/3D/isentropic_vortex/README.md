# 3D Isentropic Vortex Tutorial

This is the primary test case for the DG-MPI solver. It can be used to verify the implementation of the DG discritization for the Euler equations as well as be used for testing the multi-node scaling via MPI.

## Step 1: Download the meshes

The meshes for this case are stored on google drive. To automatically download the meshes use the following commands (assuming you are in the `isentropic_vortex` directory:

`cd meshes`
`python download_mesh_pkg.py`

