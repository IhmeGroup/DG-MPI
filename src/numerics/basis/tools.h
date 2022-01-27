#ifndef DG_NUMERICS_BASIS_TOOLS_H
#define DG_NUMERICS_BASIS_TOOLS_H

#include "common/defines.h"
#include "common/my_exceptions.h"

#include <Kokkos_Core.hpp>

namespace BasisTools {

void equidistant_nodes_1D_range(rtype start, rtype stop, int nnodes,
	Kokkos::View<rtype*> &xnodes);


} // end namespace BasisTools

#endif // DG_NUMERICS_BASIS_TOOLS_H