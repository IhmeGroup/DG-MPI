#ifndef DG_SOLVER_HELPERS_H
#define DG_SOLVER_HELPERS_H

#include <Kokkos_Core.hpp>

namespace VolumeHelpers {

struct VolumeHelperFunctor {

	~VolumeHelperFunctor() = default;

	KOKKOS_FUNCTION
    void operator()(const int ie) const;

};

} // end namespace VolumeHelper

#endif // DG_SOLVER_HELPERS_H