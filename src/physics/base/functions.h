#ifndef DG_PHYSICS_BASE_FUNCTIONS_H
#define DG_PHYSICS_BASE_FUNCTIONS_H

#include "common/defines.h"
#include "physics/base/data.h"

#include <Kokkos_Core.hpp>
#include "KokkosBlas1_nrm2.hpp"
#include "KokkosBlas1_axpby.hpp"
#include "KokkosBlas1_scal.hpp"


namespace BaseFcnType {

/*
Enum class that stores the types of analytical functions for initial
conditions, exact solutions, and/or boundary conditions. These
functions are generalizable to different kinds of physics.
*/
enum class FcnType {
    Uniform,
    None,
};

/*
---------------
State functions
---------------
These classes inherit from the FcnBase class. See FcnBase for detailed
comments of attributes and methods. Information specific to the
corresponding child classes can be found below. These classes should
correspond to the FcnType enum members above.
*/

/*
This class sets a uniform state.\
*/
class Uniform: public FcnBase {
    public:
        /*
        Constructor sets read in state

        Inputs:
        -------
            physics: physics object
        Outpus:
        -------
            state: input condition for uniform condition [ns]
        */
        template<int dim>
        Uniform(Physics::PhysicsBase<dim> &physics, const rtype *state);

        /*
        Destructor
        */
        ~Uniform();

        /*
        Set Uniform state as given x-coordinates and time

        This function takes the stored state from the physics class and sets the values
        in the solvers state

        Inputs:
        -------
            physics: physics object
            x: coordinates in physical space [ndims]
            t: time
        */
        template<int dim> DG_KOKKOS_FUNCTION
        void get_state(Physics::PhysicsBase<dim> &physics, const rtype *x, const rtype *t,
            Kokkos::View<rtype*> Uq);
};

} // end namespace BaseFcnType


/*
------------------------
Numerical flux functions
------------------------
These classes inherit from the ConvNumFluxBase or DiffNumFluxBase class.
See ConvNumFluxBase/DiffNumFluxBase for detailed comments of attributes
and methods. Information specific to the corresponding child classes can
be found below. These classes should correspond to the ConvNumFluxType
or DiffNumFluxType enum members.
*/
namespace BaseConvNumFluxType {

/*
Enum class that stores the types of convective numerical fluxes. These
numerical fluxes are generalizable to different kinds of physics.
*/
enum class ConvNumFluxType {
    LaxFriedrichs,
    None,
};

/*
This class corresponds to the local Lax-Friedrichs flux function
*/
class LaxFriedrichs: public ConvNumFluxBase {
    public:
        template<int dim> DG_KOKKOS_FUNCTION
        void compute_flux(Physics::PhysicsBase<dim> &physics,
            Kokkos::View<rtype*> UqL,
            Kokkos::View<rtype*> UqR,
            Kokkos::View<rtype*> normals,
            Kokkos::View<rtype*> Fq);
};

} // end namespace BaseConvNumFluxType

#endif //DG_PHYSICS_BASE_FUNCTIONS_H
