#include "common/defines.h"
#include "physics/base/data.h"
#include "physics/base/functions.h"


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

    //NEEDS TO BE IMPLEMENTED
};

} // end namespace BaseConvNumFluxType
