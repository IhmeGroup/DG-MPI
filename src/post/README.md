# `post`

The post module is designed as a separate executable file. Users should follow the [isentropic vortex tutorial](https://github.com/IhmeGroup/DG-MPI/tree/main/examples/euler/3D/isentropic_vortex) to use the `post` executable.

Currently, `post` mimics the start-up of the solver's executable by constructing all the helper functions and data types. This could be modified in the future, but in this way, the same functions that are used for the solver are used here.
Essentially, there is currently one function in `post`. The calculation of the L2-norm error of a case. Given that the case you are running has an exact solution, this function will calculate the L2 error. This is very useful for grid convergence studies.

Future work should be to implement plotting functions within this executable.
