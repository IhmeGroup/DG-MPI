#ifndef DG_SOLVER_HELPERS_H
#define DG_SOLVER_HELPERS_H

#include "numerics/basis/basis.h"
#include <Kokkos_Core.hpp>

namespace VolumeHelpers {

struct VolumeHelperFunctor {

	// ~VolumeHelperFunctor() = default;
	VolumeHelperFunctor(Basis::Basis basis);

	KOKKOS_FUNCTION
    void operator()(const int ie) const;

 	void get_quadrature(Basis::Basis base, 
 		const int order);

    Kokkos::View<rtype**> quad_pts;
    Kokkos::View<rtype*> quad_wts;
};

} // end namespace VolumeHelper

#include "solver/helpers.cpp"

#endif // DG_SOLVER_HELPERS_H