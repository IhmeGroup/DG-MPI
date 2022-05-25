## `exec` 

The `exec` directory is the so-called "engine" of the solver. This is the first thing executed and is also the location of the solver's executable file once 
the code is compiled. The primary purpose of this directory and the `main.cpp` file is to instantiate the needed objects for the solver and run it. 

Some specific things to note related to this directory:

1. This is where the memory network is instantiated. This is where `MPI_Init` and `Kokkos::initialize` are called.
2. We use `toml` for parsing the input deck. This is a simple and powerful tool. Feel free to read more about toml [here](https://marzer.github.io/tomlplusplus/)
3. The entire code is templated on the number of dimensions. Currently, the solver supports both two and three dimensions. We choose to not support 1D explicitly as the lab uses [quail](https://github.com/IhmeGroup/quail) for any 1D cases. This is where the dimension is set prior to any execution of the code.
5. The geometric order is HARD CODED to one in this directory. This can be modified by reading the geometric order from the `gmsh` file. The solver should support curved meshes, but this has not been tested.
6. The primary functions of the code are then called here. These include:
    1. Instantiation of the mesh object
    2. Instantiation of the solver object
    3. Precomputing the helper objects
    4. Initializing the solution coefficients
    5. Running the solver
    6. Write the solution to disk
