![C++](https://img.shields.io/badge/C%2B%2B-17-blue)
![GitHub](https://img.shields.io/github/license/IhmeGroup/DG-MPI)


# DG-MPI

This repository is currently in development. As of 5/17/2022 the current capabilities consists of a Discontinuous Galerkin solver that can solve the Euler equations in 3D on periodic grids. This code targets performance portability and can be run on NVIDIA GPUs or any CPU (In theory, it should also work with AMD GPUs, but this has not been tested).

# [`How to Build`](https://github.com/IhmeGroup/DG-MPI/tree/main/src/others)

The build process is well defined for the Yellowstone HPC system at Stanford University. Users are directed to the following [`scripts`](https://github.com/IhmeGroup/DG-MPI/tree/main/src/others) for the build procedure.
<!-- First, get the dependencies. This is done by running
    ```
    git submodule init
    ```
then following the instructions in [`externals`](https://github.com/IhmeGroup/DG-MPI/tree/main/externals).

To compile the solver, run:
```
mkdir build
cd build
cmake ..
make
``` -->

## Style Guide
https://google.github.io/styleguide/cppguide.html
