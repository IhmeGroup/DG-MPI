# `solver`

The solver module is the workhorse module for the entire code. This module includes the solver object (in `base.h` and `base.cpp`), the volume and interior face flux functors (in `flux_functors_impl.h`), and the primary helper objects for the volume and face residual functions (in `helpers.h` and `helpers.cpp`).

## Solver object

The solver object defined in `base.h` is the primary object for the entire code. It contains the `Uc` view, which defines the solution coefficients, the `physics`, `basis`, `stepper`, `vol_helpers`, `mesh`, `network`, and `iface_helpers` objects. If you have access to the solver object, you essentially have access to any data in the code.

The most important functions in the solver object include `get_element_residuals`, `get_interior_face_residuals`, and several other functions that are used in calculating the residuals.

## Flux Functors

The flux functors (that are defined as Kokkos kernels) are defined in `flux_functors_impl.h`. We have two functors, the volume flux functor and the interior face flux functor. These kernels are launched on the CPU/GPU for each quadrature point in the solution space. 

Note: These are designed for convective fluxes currently. There are several commented sections that can be adapted for diffusion fluxes if desired. These are not tested and should be tested explicitly before being used.

## Helper objects

We have two helper objects. The volume and iface helper objects. These objects contain precomputed information that is stored and used in the residual calculations during each time step. The idea is to not recompute things like basis functions or quadrature points/weights unnecessarily. The members of these objects can be accessed anywhere the user has access to the `solver` object.
