// #include "physics/base/base.h"
// #include "physics/euler/data.h"
#include "physics/euler/functions.h"
#include <cmath>

namespace Physics {

// using FcnMap = std::map<std::string, int>;

// void process_map(std::string fcn_type, FcnMap fcn_map)

/* --------------------------------------------------------
    Physics Method Definitions + Function Pointer Wrappers
----------------------------------------------------------*/
template<unsigned dim> inline
Physics<dim>::Physics(PhysicsType physics_type, 
    std::string _IC_name) : physics_type{physics_type} {
    
    // set the initial condition enum
    IC_type = enum_from_string<ICType>(_IC_name.c_str());
}


template<unsigned dim> KOKKOS_INLINE_FUNCTION
void Physics<dim>::get_conv_flux_interior(const rtype* U, const rtype* gU, 
    rtype* F, rtype* Fdir) const {

    if (physics_type == PhysicsType::Euler){
        EulerFcnType::conv_flux_interior<dim>(U, Fdir);
    }
    else {
        printf("ERROR: NO FLUX FUNCTION\n"); // TODO: Figure out throws on GPU
    }

}

template<>
template<typename ViewTypeX, typename ViewTypeUq> KOKKOS_INLINE_FUNCTION
void Physics<2>::call_IC(ViewTypeX x, const rtype t,
        ViewTypeUq Uq) const {
    if (IC_type == ICType::Uniform){
        set_state_uniform_2D(this, x, t, Uq);
    }
    if (IC_type == ICType::Gaussian){
        set_gaussian_state_2D(this, x, t, Uq);
    }
}

template<>
template<typename ViewTypeX, typename ViewTypeUq> KOKKOS_INLINE_FUNCTION
void Physics<3>::call_IC(ViewTypeX x, const rtype t,
        ViewTypeUq Uq) const {
    if (IC_type == ICType::Sphere){
        set_smooth_sphere(this, x, t, Uq);
    }
}

template<>
template<typename ViewTypeX, typename ViewTypeUq> KOKKOS_INLINE_FUNCTION
void Physics<2>::call_exact_solution(ViewTypeX x, const rtype t,
        ViewTypeUq Uq) const {
    if (IC_type == ICType::Uniform){
        set_state_uniform_2D(this, x, t, Uq);
    }
    else if (IC_type == ICType::Gaussian){
        set_gaussian_state_2D(this, x, t, Uq);
    } else {
        printf("THERE IS NO EXACT SOLUTION PROVIDED");
    }
}

template<>
template<typename ViewTypeX, typename ViewTypeUq> KOKKOS_INLINE_FUNCTION
void Physics<3>::call_exact_solution(ViewTypeX x, const rtype t,
        ViewTypeUq Uq) const {
    
    if (IC_type == ICType::Sphere){
        set_smooth_sphere(this, x, t, Uq);
    } else if (IC_type == ICType::Uniform){
        set_state_uniform_3D(this, x, t, Uq);
    } else {
        printf("THERE IS NO EXACT SOLUTION PROVIDED");
    }
}


template<typename ViewTypeX, typename ViewTypeUq> KOKKOS_INLINE_FUNCTION
void set_state_uniform_2D(const Physics<2>* physics, ViewTypeX x, const rtype t,
        ViewTypeUq Uq){

    for (long unsigned is = 0; is < Uq.extent(0); is++){
        Uq(is) = physics->IC_data[is];
    }

}

template<typename ViewTypeX, typename ViewTypeUq> KOKKOS_INLINE_FUNCTION
void set_state_uniform_3D(const Physics<3>* physics, ViewTypeX x, const rtype t,
        ViewTypeUq Uq){

    for (long unsigned is = 0; is < Uq.extent(0); is++){
        Uq(is) = physics->IC_data[is];
    }

}

template<typename ViewTypeX, typename ViewTypeUq> KOKKOS_INLINE_FUNCTION
void set_gaussian_state_2D(const Physics<2>* physics, ViewTypeX x, const rtype t,
        ViewTypeUq Uq){

    rtype A;
    for (unsigned is = 0; is < physics->NUM_STATE_VARS; is++){
        A = physics->IC_data[is];

        // rtype xdir = x(0)*x(0) / (2.*0.2*0.2);
        // rtype ydir = x(1)*x(1) / (2.*0.4*0.4);

        Uq[is] = A * sin(30. * x(0)*x(0)*x(1)) * (x(0) + 1.)*(x(1) + 1.);
    }

}

template<typename ViewTypeX, typename ViewTypeUq> KOKKOS_INLINE_FUNCTION
void set_smooth_sphere(const Physics<3>* physics, ViewTypeX x, const rtype t,
        ViewTypeUq Uq){

    rtype A;
    const rtype v1 = 1.;
    const rtype v0 = 0.;
    const rtype R = 0.5;
    const rtype w = 0.5;

    rtype magR = sqrt(x(0)*x(0) + x(1)*x(1) + x(2)*x(2));

    for (unsigned is = 0; is < physics->NUM_STATE_VARS; is++){
        A = physics->IC_data[is];

        Uq[is] = 0.5 * (v1 - v0) * (tanh(M_PI * (R - magR)/w) + 1.0) + v0;
    }
}
// 0.000190
// 0.000241
    
// template<int dim> int PhysicsBase<dim>::get_NS(){
//     return 0;
// }

// template <int dim> DG_KOKKOS_FUNCTION
// void PhysicsBase<dim>::get_conv_flux_projected(
//     Kokkos::View<const rtype*> Uq,
//     Kokkos::View<const rtype*> normals,
//     Kokkos::View<rtype*> Fproj){
//     // unpack
//     int NS = get_NS();

//     // allocate view for directional flux
//     Kokkos::View<rtype**> Fq("Fq", NS, dim);
//     conv_flux_physical(Uq, Fq);
//     KokkosBlas::gemv ("N", 1., Fq, normals, 1., Fproj);
// }

// template<int dim> DG_KOKKOS_FUNCTION
// void PhysicsBase<dim>::conv_flux_physical(
//     Kokkos::View<const rtype*> U,
//     Kokkos::View<rtype**> F){
//     // throw NotImplementedException("PhysicsBase does not implement "
//  //                                      "conv_flux_physical -> implement in child class");
// }

// template<int dim> DG_KOKKOS_FUNCTION
// rtype PhysicsBase<dim>::get_maxwavespeed(Kokkos::View<const rtype*> U) {
// // throw NotImplementedException("PhysicsBase does not implement "
// //                                       "get_maxwavespeed -> implement in child class");
// }


// template<int dim> DG_KOKKOS_FUNCTION
// rtype PhysicsBase<dim>::compute_variable(std::string str,
//       Kokkos::View<const rtype*> Uq){

//     var = PhysicsBase<dim>::get_physical_variable(str);
//     // var = enum_from_string<PhysicsVariables>(str.c_str());

//     // NEED TO UPDATE THIS FUNCTION TO GET SPECIFIC VAR FUNCTIONS
//     return 0.0;
// }

// template class PhysicsBase<2>;
// template class PhysicsBase<3>;
} // end namespace Physics
