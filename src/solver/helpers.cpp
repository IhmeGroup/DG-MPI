#include "solver/helpers.h"

namespace VolumeHelpers {

	KOKKOS_FUNCTION
    void VolumeHelperFunctor::operator()(const int ie) const {

    	 printf("Hello from ie = %i\n", ie);
    }

} // end namespace VolumeHelper