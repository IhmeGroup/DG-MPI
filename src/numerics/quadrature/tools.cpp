#include "numerics/quadrature/tools.h"

namespace QuadratureTools {

using std::ceil, std::pow;

int get_gausslegendre_quadrature_order(const int order_,
        const int NDIMS){

    // TODO: For now, we set gorder = 1. It would be better
    // to have it passed here somehow
    // 
    // we also need to discuss this with Kihiro 
    // as its unclear of the choices he made in the legion code
    const int gorder = 1;
    return 2 * (order_ + gorder) + 1;
}


void get_number_of_quadrature_points(const int order, const int NDIMS, int& nq_1d, int& nq){
    if ((order + 2) % 2 == 0) {nq_1d = (order + 2) / 2;}
    if ((order + 1) % 2 == 0) {nq_1d = (order + 1) / 2;}
    nq = pow(nq_1d, NDIMS);
}

}
