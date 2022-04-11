#ifndef DG_PHYSICS_BASE_FUNCTIONS_H
#define DG_PHYSICS_BASE_FUNCTIONS_H

#include "common/defines.h"
#include "common/enums.h"
// #include "physics/base/data.h"

#include <Kokkos_Core.hpp>
#include "KokkosBlas1_nrm2.hpp"
#include "KokkosBlas1_axpby.hpp"
#include "KokkosBlas1_scal.hpp"


// /*
// ------------------------
// Numerical flux functions
// ------------------------
// These classes inherit from the ConvNumFluxBase or DiffNumFluxBase class.
// See ConvNumFluxBase/DiffNumFluxBase for detailed comments of attributes
// and methods. Information specific to the corresponding child classes can
// be found below. These classes should correspond to the ConvNumFluxType
// or DiffNumFluxType enum members.
// */
namespace BaseConvNumFluxType {

// /*
// Enum class that stores the types of convective numerical fluxes. These
// numerical fluxes are generalizable to different kinds of physics.
// */
// enum class ConvNumFluxType {
//     LaxFriedrichs,
//     None,
// };

// /*
// This class corresponds to the local Lax-Friedrichs flux function
// */
// class LaxFriedrichs: public ConvNumFluxBase {
//     public:
//         template<int dim> DG_KOKKOS_FUNCTION
//         void compute_flux(Physics::PhysicsBase<dim> &physics,
//             Kokkos::View<rtype*> UqL,
//             Kokkos::View<rtype*> UqR,
//             Kokkos::View<rtype*> normals,
//             Kokkos::View<rtype*> Fq);
// };

} // end namespace BaseConvNumFluxType



#endif //DG_PHYSICS_BASE_FUNCTIONS_H
