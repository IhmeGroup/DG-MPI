#include "physics/base/data.h"

template<int dim>
void FcnBase::get_state(Physics::PhysicsBase<dim> &physics, const rtype *x, const rtype *t){
	throw NotImplementedException("FcnBase does not implement "
									"get_state -> implement in child class");
}

template<int dim>
void ConvNumFluxBase::compute_flux(Physics::PhysicsBase<dim> &physics,             
		Kokkos::View<rtype*> UqL,
        Kokkos::View<rtype*> UqR, 
        Kokkos::View<rtype*> normals,
        Kokkos::View<rtype*> Fq){
	throw NotImplementedException("ConvNumFluxBase does not implement "
									"compute_flux -> implement in child class");
}