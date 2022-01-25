#ifndef DG_PHYSICS_BASE_FUNCTIONS_H
#define DG_PHYSICS_BASE_FUNCTIONS_H

#include "common/defines.h"
#include "physics/base/data.h"

#include <Kokkos_Core.hpp>

namespace BaseFcnType {

enum class FcnType {
	Uniform,
	None,
};


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
    template<unsigned dim>
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
    template<unsigned dim>
    void get_state(Physics::PhysicsBase<dim> &physics, const rtype *x, const rtype *t,
    	Kokkos::View<rtype*> Uq);
};

} // end namespace BaseFcnType


#endif //DG_PHYSICS_BASE_FUNCTIONS_H