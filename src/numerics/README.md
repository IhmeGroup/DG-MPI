# `numerics`

The numerics module consists of three major submodules: The basis functions, the quadrature rules, and the time steppers.

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
3. Add the construction of the new basis type in the `Basis` constructor function in `numerics/basis.cpp`. For example, the user could follow the exact template provied by Legendre polynomial basis fnctions:
    ```
    if (basis_type == BasisType::LegendreQuad){
        get_values_pointer = get_values_legendrequad;
        get_grads_pointer = get_grads_legendrequad;
        name = "LegendreQuad";
        shape = Shape(enum_from_string<ShapeType>("Quadrilateral"));
        face_shape = Shape(enum_from_string<ShapeType>("Segment"));

    }
    ```

