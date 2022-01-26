#include "common/defines.h"
#include "physics/base/data.h"
#include "physics/base/functions.h"

#include "KokkosBlas1_nrm2.hpp"


namespace BaseFcnType {

template<int dim>
Uniform::Uniform(Physics::PhysicsBase<dim> &physics, const rtype *state){

    int NS = physics.get_NS();
    for (int is = 0; is < NS; is++){
        physics->state[is] = state[is];
    }
}

template<int dim> DG_KOKKOS_FUNCTION
void Uniform::get_state(Physics::PhysicsBase<dim> &physics, const rtype *x, const rtype *t, 
    Kokkos::View<rtype*> Uq){
    
    int NS = physics.get_NS();
    for (int is = 0; is < NS; is++){
        Uq(is) = physics->state[is];
    }
}

} // end namespace BaseFcnType


namespace BaseConvNumFluxType {

template<int dim> DG_KOKKOS_FUNCTION
void LaxFriedrichs::compute_flux(Physics::PhysicsBase<dim> &physics, 
            Kokkos::View<rtype*> UqL,
            Kokkos::View<rtype*> UqR, 
            Kokkos::View<rtype*> normals,
            Kokkos::View<rtype*> Fq){

    // unpack
    int NS = physics.get_NS();

    // Normalize the normal vectors
    Kokkos::View<rtype*> n_hat("n_hat", dim);
    rtype n_mag = KokkosBlas::nrm2(normals);
    for (int i = 0; i < dim; i++){
        n_hat(i) = normals(i)/n_mag;
    }

    // Left flux
    Kokkos::View<rtype*> FqL("FqL", NS);
    FqL = physics.get_conv_flux_projected(UqL, n_hat);

    // Right flux
    Kokkos::View<rtype*> FqR("FqR", NS);
    FqR = physics.get_conv_flux_projected(UqR, n_hat);

    // Jump condition
    Kokkos::View<rtype*> dUq("dUq", NS);
    // dUq = UqR - UqL;

    // Calculate the max wave speed
    rtype a = physics.compute_variable("MaxWaveSpeed", UqL);
    rtype aR = physics.compute_variable("MaxWaveSpeed", UqR);





};

} // end namespace BaseConvNumFluxType
