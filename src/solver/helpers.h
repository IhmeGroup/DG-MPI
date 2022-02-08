#ifndef DG_SOLVER_HELPERS_H
#define DG_SOLVER_HELPERS_H

#include <Kokkos_Core.hpp>

namespace VolumeHelpers {

struct VolumeHelperFunctor {

	// ~VolumeHelperFunctor() = default;
	VolumeHelperFunctor();

	KOKKOS_FUNCTION
    void operator()(const int ie) const;


    int testing;
};

} // end namespace VolumeHelper

#include "solver/helpers.cpp"

#endif // DG_SOLVER_HELPERS_H