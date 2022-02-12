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


void get_number_of_quadrature_points(const int order, const int NDIMS, int& nq_1d, int& nq){

    if ((order + 2) % 2 == 0) {nq_1d = (order + 2) / 2;}
    if ((order + 1) % 2 == 0) {nq_1d = (order + 1) / 2;}
    nq = pow(nq_1d, NDIMS);
}

}