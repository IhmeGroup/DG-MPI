#ifndef DG_NUMERICS_QUADRATURE_TOOLS_H
#define DG_NUMERICS_QUADRATURE_TOOLS_H

#include "common/defines.h"
#include "common/my_exceptions.h"
#include <math.h>

#include <Kokkos_Core.hpp>

namespace QuadratureTools {

int get_gausslegendre_quadrature_order(const int order_, 
		const int NDIMS);

void get_number_of_quadrature_points(const int order, const int NDIMS, 
		int& nq_1d, int& nq);

} // end namespace QuadratureTools


#endif // DG_NUMERICS_QUADRATURE_TOOLS_H