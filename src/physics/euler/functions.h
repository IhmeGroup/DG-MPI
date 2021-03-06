#ifndef DG_PHYSICS_EULER_FUNCTIONS_H
#define DG_PHYSICS_EULER_FUNCTIONS_H

#include "common/defines.h"
// #include "physics/base/base.h" 
#include <Kokkos_Core.hpp>
#include "math/linear_algebra.h"
#include "physics/base/base.h"

// // #include "KokkosBlas1_nrm2.hpp"
// // #include "KokkosBlas1_axpby.hpp"
// // #include "KokkosBlas1_scal.hpp"


namespace EulerFcnType {

template<unsigned dim> KOKKOS_INLINE_FUNCTION
rtype get_pressure(const rtype& gamma, const rtype* U);
// template<int dim> DG_KOKKOS_FUNCTION
// rtype Euler<dim>::get_maxwavespeed(Kokkos::View<const rtype*> U) {
//     // unpack
//     auto mom = Kokkos::subview(U, Kokkos::make_pair(1, dim + 1));
//     auto rho1 = 1./U(0);

//     return KokkosBlas::nrm2(mom) * rho1 + sqrt(gamma * get_pressure(U) * rho1);

// }

template<unsigned dim> KOKKOS_INLINE_FUNCTION
rtype get_maxwavespeed(const rtype& gamma, const rtype* U);

template<unsigned dim> KOKKOS_INLINE_FUNCTION
void conv_flux_interior(const rtype& gamma, const rtype* U,
    rtype* Fdir);

template<typename ViewTypeX, typename ViewTypeUq> KOKKOS_INLINE_FUNCTION
void set_state_isentropic_vortex(const Physics::Physics<2>* physics, ViewTypeX x, const rtype t,
    ViewTypeUq Uq);

// /*
// ---------------
// State functions
// ---------------
// These classes inherit from the FcnBase class. See FcnBase for detailed
// comments of attributes and methods. Information specific to the
// corresponding child classes can be found below. These classes should
// correspond to the FcnType enum members above.
// */

// /*
// This function sets a uniform state.\
// */
// // template<typename ViewType>
// // KOKKOS_INLINE_FUNCTION
// // void set_state_uniform_2D(const Physics::Physics& physics, scratch_view_1D_rtype x, const rtype t, 
// //     scratch_view_1D_rtype Uq);
// // KOKKOS_INLINE_FUNCTION
// inline
// void set_state_uniform_2D();

// // class Uniform: public FcnBase {
// //     public:
// //         /*
// //         Constructor sets read in state

// //         Inputs:
// //         -------
// //             physics: physics object
// //         Outpus:
// //         -------
// //             state: input condition for uniform condition [ns]
// //         */
// //         template<int dim>
// //         Uniform(Physics::PhysicsBase<dim> &physics, const rtype *state);

// //         /*
// //         Destructor
// //         */
// //         ~Uniform();

        
// //         Set Uniform state as given x-coordinates and time

// //         This function takes the stored state from the physics class and sets the values
// //         in the solvers state

// //         Inputs:
// //         -------
// //             physics: physics object
// //             x: coordinates in physical space [ndims]
// //             t: time
        
// //         template<int dim> DG_KOKKOS_FUNCTION
// //         void get_state(Physics::PhysicsBase<dim> &physics, const rtype *x, const rtype *t,
// //             Kokkos::View<rtype*> Uq);
// // };

} // end namespace EulerFcnType

namespace EulerConvNumFluxType {

template<unsigned dim> KOKKOS_INLINE_FUNCTION
void compute_flux_hllc(const Physics::Physics<dim>& physics,
    const rtype* UL, const rtype* UR, const rtype* N, 
    rtype* F, rtype* gUL, rtype* gUR);
    
}

#include "physics/euler/functions.cpp"

#endif // end DG_PHYSICS_EULER_FUNCTIONS