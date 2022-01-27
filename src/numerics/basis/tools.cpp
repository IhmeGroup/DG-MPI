#include "numerics/basis/tools.h"

namespace BasisTools {

DG_KOKKOS_FUNCTION
void equidistant_nodes_1D_range(rtype start, rtype stop, int nnodes,
	Kokkos::View<rtype*> &xnodes) {

	if (nnodes <= 1){
		throw ValueErrorException("Need at least two nodes to compute");
	}
	if (stop <= start) {
		throw ValueErrorException("Assume beginning is smaller than end");
	}
	if (xnodes.extent (0) != nnodes){
		Kokkos::resize(xnodes, nnodes);
	}

	rtype dx = (stop - start) / ((rtype)nnodes - 1.);

	for (int i = 0; i < nnodes; i++){
		xnodes(i) = start + (rtype)i * dx;
	}
} 

} // end namespace BasisTools