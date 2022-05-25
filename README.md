![C++](https://img.shields.io/badge/C%2B%2B-17-blue)
![GitHub](https://img.shields.io/github/license/IhmeGroup/DG-MPI)


# DG-MPI

This repository is currently in development. As of 5/17/2022 the current capabilities consists of a Discontinuous Galerkin solver that can solve the Euler equations in 3D on periodic grids. This code targets performance portability and can be run on NVIDIA GPUs or any CPU (In theory, it should also work with AMD GPUs, but this has not been tested).

# [`How to Build`](https://github.com/IhmeGroup/DG-MPI/tree/main/src/others)

The build process is well defined for the Yellowstone HPC system at Stanford University. Users are directed to the following [`scripts`](https://github.com/IhmeGroup/DG-MPI/tree/main/src/others) for the build procedure.

## Source code discussions / documentation

In this section, we point users to each specific module in the code. In each module, there is a brief overview of the current tools in the module plus the design choices. These choices are discussed and details are provided for some next steps and how to use the module. Each module is a subdirectory under `src/`.

1. [common](https://github.com/IhmeGroup/DG-MPI/tree/main/src/common)
2. [converter](https://github.com/IhmeGroup/DG-MPI/tree/main/src/converter)
3. [exec](https://github.com/IhmeGroup/DG-MPI/tree/main/src/exec)
4. [io](https://github.com/IhmeGroup/DG-MPI/tree/main/src/io)
5. [math](https://github.com/IhmeGroup/DG-MPI/tree/main/src/math)
6. [memory](https://github.com/IhmeGroup/DG-MPI/tree/main/src/memory)
7. [mesh](https://github.com/IhmeGroup/DG-MPI/tree/main/src/mesh)
8. [numerics](https://github.com/IhmeGroup/DG-MPI/tree/main/src/exec)
9. [others](https://github.com/IhmeGroup/DG-MPI/tree/main/src/others)
10. [physics](https://github.com/IhmeGroup/DG-MPI/tree/main/src/physics)
11. [post](https://github.com/IhmeGroup/DG-MPI/tree/main/src/post)
12. [solver](https://github.com/IhmeGroup/DG-MPI/tree/main/src/solver)
13. [utils](https://github.com/IhmeGroup/DG-MPI/tree/main/src/utils)
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
