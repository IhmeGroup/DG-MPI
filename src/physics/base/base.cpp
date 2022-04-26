// #include "physics/base/base.h"
// #include "physics/euler/data.h"
#include "physics/base/functions.h"
#include "physics/euler/functions.h"
#include <cmath>

namespace Physics {

// using FcnMap = std::map<std::string, int>;

// void process_map(std::string fcn_type, FcnMap fcn_map)

/* --------------------------------------------------------
    Physics Method Definitions + Function Pointer Wrappers
----------------------------------------------------------*/
template<unsigned dim> inline
Physics<dim>::Physics(PhysicsType physics_type, NumericalFluxType numerical_flux_type,
    std::string _IC_name) : physics_type{physics_type}, numerical_flux_type{numerical_flux_type} {
    
    // set the initial condition enum
    IC_type = enum_from_string<ICType>(_IC_name.c_str());
}

template<unsigned dim> inline
void Physics<dim>::set_physical_params(const toml::value& physics_params){
 
    if (physics_type == PhysicsType::Euler){
        gamma = toml::find_or<rtype>(physics_params, "gamma", 1.4);
        R = toml::find_or<rtype>(physics_params, "GasConstant", 287.05);
    }

}


template<unsigned dim> KOKKOS_INLINE_FUNCTION
void Physics<dim>::get_conv_flux_interior(const rtype* U, const rtype* gU, 
    rtype* F, rtype* Fdir) const {

    if (physics_type == PhysicsType::Euler){
        EulerFcnType::conv_flux_interior<dim>(gamma, U, Fdir);
    }
    else {
        printf("ERROR: NO FLUX FUNCTION\n"); // TODO: Figure out throws on GPU
    }

}

template<unsigned dim> KOKKOS_INLINE_FUNCTION
void Physics<dim>::get_conv_flux_numerical(const rtype* UL, const rtype* UR, 
        const rtype* N, rtype* F, rtype* gUL, rtype* gUR) const {
        
    if (numerical_flux_type == NumericalFluxType::LaxFriedrichs){
        BaseConvNumFluxType::compute_flux_laxfriedrichs(*this, UL, UR, N, F, gUL, gUR);
    }
    else if (numerical_flux_type == NumericalFluxType::HLLC){
        EulerConvNumFluxType::compute_flux_hllc(*this, UL, UR, N, F, gUL, gUR);
    }
    else {
        printf("ERROR: NO NUMERICAL FLUX FUNCTION\n"); // TODO: Figure out throws on GPU
    }

}

template<unsigned dim> KOKKOS_INLINE_FUNCTION
rtype Physics<dim>::get_maxwavespeed(const rtype* U) const {

    if (physics_type == PhysicsType::Euler){
        return EulerFcnType::get_maxwavespeed<dim>(gamma, U);
    }
    else {
        printf("ERROR: NO MAX WAVE SPEED\n"); // TODO: Figure out throws on GPU
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
    if (IC_type == ICType::IsentropicVortex){
        EulerFcnType::set_state_isentropic_vortex(this, x, t, Uq);
    }
}

template<>
template<typename ViewTypeX, typename ViewTypeUq> KOKKOS_INLINE_FUNCTION
void Physics<3>::call_IC(ViewTypeX x, const rtype t,
        ViewTypeUq Uq) const {
    if (IC_type == ICType::Sphere){
        set_smooth_sphere(this, x, t, Uq);
    }
    if (IC_type == ICType::Uniform){
        set_state_uniform_3D(this, x, t, Uq);
    }
    if (IC_type == ICType::IsentropicVortex){
        EulerFcnType::set_state_isentropic_vortex(this, x, t, Uq);
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
    } 
    else if (IC_type == ICType::IsentropicVortex){
        EulerFcnType::set_state_isentropic_vortex(this, x, t, Uq);
    }
    else {
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
    } else if (IC_type == ICType::IsentropicVortex){
        EulerFcnType::set_state_isentropic_vortex(this, x, t, Uq);
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

        //TODO: Change this back to  a gaussian function
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

} // end namespace Physics
