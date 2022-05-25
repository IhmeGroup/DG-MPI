# `numerics`

The numerics module consists of three major submodules: The basis functions, the quadrature rules, and the time steppers. This format follows a similar format to the teaching and prototyping tool [quail](https://github.com/IhmeGroup/quail/). Users are encouraged to look at quail for implementation details of specific numercs. 

## Basis

Basis is broken up into three separate parts. `basis.h`, `shape.h`, and `tools.h`.

### `basis.h`

The basis class utilizes function pointers to construct the basis object and its members. Originally, it was thought that function pointers would work better for 
GPUs than classical object oriented programming. It turns out that both suffer from the so-called "V-table" problems. Essentially, whenever you enter into a GPU kernel
an object's members need to be known. If they were defined (either by function pointers or by inheritance) outside of the kernel, then the GPU has no idea which specific
function you are attempting to call. The Kokkos tutorials give one way of approaching this by constructing objects inside of the GPU kernel (which has some amount of overhead)
allowing them to have some amount of inheritence. A user could also define the function pointers inside of the GPU kernel. All this to be said, we chose to write things 
in terms of function pointers and we essentially "roll our own" inheritance through this. 

For the basis class we have support for the following basis functions:

1. LagrangeSeg
2. LagrangeQuad
3. LagrangeHex
4. LegendreSeg
5. LegendeQuad
6. LegendreHex

If a user wanted to implement a new basis function (lets call it `UserBasis`) they would follow these steps:

1. In `common/enums.h` they would need to add `UserBasis` to `enum BasisType` in `common/enums.h` as well as adding the corresponding string name of the basis type.
2. In `numerics/basis.h` and `numerics/basis.cpp` they would need to add `get_values_userbasis` and a `get_grads_userbasis` functions. Users are encouraged to follow 
the examples from both the Lagrange and Legendre basis functions.
3. Add the construction of the new basis type in the `Basis` constructor function in `numerics/basis.cpp`. For example, the user could follow a template like this:
    ```
    if (basis_type == BasisType::UserBasis){
        get_values_pointer = get_values_userbasis;
        get_grads_pointer = get_grads_userbasis;
        name = "UserBasis";
        shape = Shape(enum_from_string<ShapeType>("<shape of user basis>"));
        face_shape = Shape(enum_from_string<ShapeType>("<face shape of user basis>"));
    }
    ```
### `shape.h`

The shape file includes the definition of the shape object. This object is instantiated by the instantiation of the basis class. Therefore, if a user is uning a 2D Lagrange simulation, the shape that is instantiated will be the `Quadrilateral` shape. Currently, the following shapes are supported:

1. Segment
2. Quadrilateral
3. Hexahedron

Additional shapes can be added. For example, if a user wanted to add triangles they would follow a similar group of steps as the basis functions described above. Templates for constructing shapes exist in the `shape.h` file. A triangular basis would follow the same format as the others, but with specific functions for its shape. (one could use the definitions for a triangular shape [here](https://github.com/IhmeGroup/quail/blob/main/src/numerics/basis/basis.py)). 


### `tools.h` 

Anytime a "tools" file is named it means that there are stand alone functions that can be called by the objects. These functions are separated from the basis or shape files because they can be valid for multiple instantiations of the basis/shape classes. 

## Quadrature

The current solver only uses Gauss Legendre quadrature rules. These can be extended to Gauss Lobatto or Dunavant (for triangles) as the need arises. The enum file currently has enums for these other quadrature rules and their implementation can be facilitated through the namespaces provided for each shape. Quadrature is defined for each specific shape (i.e. segment, quadralateral, and hexahedron). The primary way to add additional quadrature rules would be to add them within each shape file in `quadrature/<shape_name>.h`.

## Timestepping

The time steppers utilize inheritance to define new methods. Their functions exist outside of the GPU kernels. Their functions will often call solver object functions that operate on the kernels. Therefore, these methods can exist ouside of the implementation needs of the GPU. Currently we have the following time stepping schemes:

1. Forward Euler
2. 4th-order Runge-Kutta

Adding new stepper classes would follow these steps:

1. Add the stepper class name to `common/enums.h`.
2. Add the new stepper class definition in `numerics/timestepping/stepper.h`. Users can directly follow one of the current examples list in this file. The most important function is `take_time_step`! 
3. Add the stepper class to the StepperFactory. The stepper factory instantiates the stepper class directly on the heap. This then results in the use of a `shared_ptr` for the stepper class as an object stored in the `solver` object. 
