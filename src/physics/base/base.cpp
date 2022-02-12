#include "physics/base/base.h"

namespace Physics{

template<int dim> int PhysicsBase<dim>::get_NS(){
    return 0;
}

template <int dim> DG_KOKKOS_FUNCTION 
void PhysicsBase<dim>::get_conv_flux_projected(
	Kokkos::View<const rtype*> Uq, 
	Kokkos::View<const rtype*> normals,
	Kokkos::View<rtype*> Fproj){
	// unpack
	int NS = get_NS();
	
	// allocate view for directional flux
	Kokkos::View<rtype**> Fq("Fq", NS, dim);
	conv_flux_physical(Uq, Fq);
	KokkosBlas::gemv ("N", 1., Fq, normals, 1., Fproj);
}

template<int dim> DG_KOKKOS_FUNCTION
void PhysicsBase<dim>::conv_flux_physical(
    Kokkos::View<const rtype*> U,
    Kokkos::View<rtype**> F){
	// throw NotImplementedException("PhysicsBase does not implement "
 //                                      "conv_flux_physical -> implement in child class");
}

template<int dim> DG_KOKKOS_FUNCTION
rtype PhysicsBase<dim>::get_maxwavespeed(Kokkos::View<const rtype*> U) {
// throw NotImplementedException("PhysicsBase does not implement "
//                                       "get_maxwavespeed -> implement in child class");
}


template<int dim> DG_KOKKOS_FUNCTION 
rtype PhysicsBase<dim>::compute_variable(std::string str, 
      Kokkos::View<const rtype*> Uq){

	var = PhysicsBase<dim>::get_physical_variable(str);
	// var = enum_from_string<PhysicsVariables>(str.c_str());

	// NEED TO UPDATE THIS FUNCTION TO GET SPECIFIC VAR FUNCTIONS
	return 0.0;
}

template class PhysicsBase<2>;
template class PhysicsBase<3>;
} // end namespace Physics
