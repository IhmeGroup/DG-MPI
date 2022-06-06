# `physics`

The physics module consists of a base module and specific equation type modules. For example, we currently have an `euler` module. A user could extend this to include something like a Navier-Stokes module.

Similar to many other modules, the physics module is header only. This is due to the performance portable nature of the solver. This could be modified in the future, but the primary consequence is in the build time of the solver.

## Base Module

The base module contains the primary physics object in `base.h` and `base.cpp`. It includes some basic getter functions as well as the primary calls to `get_conv_flux_interior`, `get_maxwavespeed`, `call_IC`, and `call_exact_solution`. These functions each use if statements to determine which specific call of the convective flux function, initial conditions, exact solution, etc... are called. You can think of these as wrapper functions that contain calls to the specific implementations for various cases or physics types. 

Note: The strategy of using if statements for providing modularity within the solver should be assessed. It is unclear if it incurs a large hit on the GPU solver. There should not be any thread divergence as the if statements in these functions will always execute in the same way each time during a single cases run.

In addition to the physics object, there are also a few initial condition functions. These should be moved to an appropriate `functions.h` and `functions.cpp` file in the `base` folder.

There is a generalized numerical flux function (the Lax-Friedrichs flux function) implemented in `functions.h` and `functions.cpp` in the `base` module. This convective numerical flux function was written generally and can be used for any equation set.

Also, there is an abundance of legacy/degraded code commented out in these files. These can be useful when implementing new features that may be contained in the commented sections. Users should feel free to peruse this section to see if there is code that could be helpful.

## Euler Module

This module contains the specific implementation for the Euler equations. The `euler.h` and `euler.cpp` files are not used and can be considered degraded.

The important files here are `functions.h` and `functions.cpp`. These contain equation specific functions for the Euler equations. For example, this is where the `get_pressure` function is defined. In addition, this is where a user would find Euler specific initial conditions and exact solutions. These functions are contained within the `EulerFcnType` namespace. 

The namespace, `EulerConvNumFluxType`, is where equation specific flux functions are defined. For example, we have the HLLC flux function defined here.

NOTE: If a user wished to implement the Navier-Stokes equations, they would have a specific namespace for the diffusive fluxes.
