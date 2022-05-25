# `common`

The common directory holds files that can be universal across the solver. It is broken up into three files:

1. `defines.h`
2. `enums.h`
3. `my_exceptions.h`

## `defines.h`

Defines can be broken into three separate parts: 

### 1. Precision

The first section defines the precision of the solver (single or double). This is almost always set to `double`, but the main consequence is that instead of using `double` 
or `float` throughout the code, developers should use `rtype`.

### 2. `constexprs`

The second section defines multiple `constexpr` variables for the code. These define maximum values for various arrays. Of primary interest is `GLOBAL_NUM_SPECIES`. 
This is a compile time constant that defines the number of species for the simulation. This was done to make the transition to multispecies Navier-Stokes more seamless.
Currently, this value is set to 1. Throughout the code, the `GLOBAL_NUM_SPECIES` parameter is used to define the `NUM_STATE_VARS` or `NS` variable such that it is a compile time constant.

### 3. Kokkos view definitions via `using`

The third section helps minimize characters in the code by introducing various definitions for long kokkos type names. For example:

`view_type_1D = Kokkos::View<rtype*>`

## `enums.h`

Enums are often useful when it comes to `if` statements throughout the code because evaluating an `if` statement for an `integer` is more affordable than
comparing two `string` types. We therefore use enums throughout the code. This file has everything that is related to enums for the solver. This includes converting
from `enum` to `string` and vice versa (details can be seen in the file which has some comments). This is also a good place to look for the various features currently
available in the solver. When adding new features, users will add names to the appropriate enum. For example, the `StepperType` enum would be where a user adds
something like the 3rd-order Strong-Stability-Preserving Runge-Kutta scheme. This would be a descent first exercise for someone implementing something new in this code.

## `my_exceptions.h`: NOT USED IN SOLVER

This code is currently not used in the DG-MPI solver. Exceptions are complicated on GPUs. CUDA does not have native exception handling therefore, some type 
of custom exception handling is needed. There is no clearly defined exception handling in this solver. This file was copied from the DG-Legion code within the Lab of Complex Fluids. 
