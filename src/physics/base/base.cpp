// #include "physics/base/base.h"
// #include "physics/euler/data.h"
// #include "physics/euler/functions.h"

namespace Physics {

// using FcnMap = std::map<std::string, int>;

// void process_map(std::string fcn_type, FcnMap fcn_map)

/* --------------------------------------------------------
    Physics Method Definitions + Function Pointer Wrappers
----------------------------------------------------------*/
inline
Physics::Physics(PhysicsType physics_type, const int dim, std::string _IC_name){

    IC_type = enum_from_string<ICType>(_IC_name.c_str());

    if (physics_type == PhysicsType::Euler and dim == 2){ 

        NUM_STATE_VARS = 4;
        // set_IC = EulerMaps::set_IC_euler2D;
        // EulerMaps::set_IC_map_euler2D();
        
    }

    if (physics_type == PhysicsType::Euler and dim == 3) {

        NUM_STATE_VARS = 5;

    }
}

template<typename ViewTypeX, typename ViewTypeUq> KOKKOS_INLINE_FUNCTION
void Physics::call_IC(ViewTypeX x, const rtype t,
        ViewTypeUq Uq) const {
    if (IC_type == ICType::Uniform){
        set_state_uniform_2D(this, x, t, Uq);
    }

}

template<typename ViewTypeX, typename ViewTypeUq> KOKKOS_INLINE_FUNCTION
void set_state_uniform_2D(const Physics* physics, ViewTypeX x, const rtype t,
        ViewTypeUq Uq){

    for (long unsigned is = 0; is < Uq.extent(0); is++){
        Uq(is) = physics->IC_data[is];
    }

}


    
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
