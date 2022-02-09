#include "numerics/quadrature/tools.h"

namespace QuadratureTools {

using std::ceil, std::pow;

int get_gausslegendre_quadrature_order(const int order_, 
		const int NDIMS){

    // quad rules for even order can integrate one order higher
    int order = order_ + (order_ + 1) % 2;
    int nq1d = ceil((order + 1.) / 2.);
    return pow(nq1d, NDIMS);
}

}