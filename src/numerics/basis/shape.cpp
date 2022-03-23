#include "numerics/basis/shape.h"
#include <vector>

namespace Basis {

/* --------------------------------------
        Segment Shape Definitions
----------------------------------------*/
int get_num_basis_coeff_segment(int p){
    return p + 1;
}


/* --------------------------------------
        Quadrilateral Shape Definitions
----------------------------------------*/
int get_num_basis_coeff_quadrilateral(int p){
    return (p + 1) * (p + 1);
}

void get_face_pts_order_wrt_orient0_quadrilateral(const int orient, const int npts,
        Kokkos::View<int*> pts_order) {

    assert(npts == (int) pts_order.extent(0));
    switch (orient) {
        case 0: {
            for (int i=0; i<npts; i++) {
                pts_order(i) = i;
            }
            break;
        }

        case 1: {
            for (int i=0; i<npts; i++) {
                pts_order(i) = npts - 1 - i;
            }
            break;
        }

        default: {
            printf("FATAL EXCEPTION IN 'get_face_pts_order_wrt_orient0_quadrilateral'");
        }
    }
}


void get_points_on_face_quadrilateral(const int face_id, const int orient, const int np,
        const Kokkos::View<rtype**>::HostMirror face_pts,
        Kokkos::View<rtype**>::HostMirror elem_pts){

    // Positions of all nodes
    const std::vector<rtype> xref_nodes{-1, 1., -1, 1.};
    const std::vector<rtype> yref_nodes{-1, -1., 1., 1.};
    std::vector<int> node(2, 0);

    // Determine which nodes are on the desired face
    switch (face_id) {
        case 0:
            node[0] = 0;
            node[1] = 1;
            break;
        case 1:
            node[0] = 1;
            node[1] = 3;
            break;
        case 2:
            node[0] = 3;
            node[1] = 2;
            break;
        case 3:
            node[0] = 2;
            node[1] = 0;
            break;
        default:
            std::string error_message = "Face ID ";
            error_message += std::to_string(face_id);
            error_message += " not available for quadrilateral";
            throw std::runtime_error(error_message);
    }

    if (orient == 1) {
        std::swap(node[0], node[1]);
    }

    rtype x0, x1, y0, y1, b0, b1;
    x0 = xref_nodes[node[0]];
    y0 = yref_nodes[node[0]];
    x1 = xref_nodes[node[1]];
    y1 = yref_nodes[node[1]];

    for (int i = 0; i < np; i++) {
        // barycentric coordinates
        b1 = (face_pts(i, 0) + 1.) / 2.;
        b0 = 1.0 - b1;

        elem_pts(i, 0) = (b0 * x0 + b1 * x1);
        elem_pts(i, 1) = (b0 * y0 + b1 * y1);
    }
}


/* --------------------------------------
        Hexahedron Shape Definitions
----------------------------------------*/
int get_num_basis_coeff_hexahedron(int p){
    return (p + 1) * (p + 1) * (p + 1);
}

Shape::Shape(ShapeType shape_type){
    if (shape_type == ShapeType::Segment){
        // set methods
        get_num_basis_coeff = get_num_basis_coeff_segment;
        get_quadrature_order_pointer =
            SegmentQuadrature::get_gausslegendre_quadrature_order;
        get_quadrature_data =
            SegmentQuadrature::get_quadrature_gauss_legendre;
        
        // set constants
        NDIMS = 1;
        NFACES = 2;
        NCORNERS = 2;
        NUM_ORIENT_PER_FACE = 1;

        name = "Segment";
    }
    if (shape_type == ShapeType::Quadrilateral){
        // set methods
        get_num_basis_coeff = get_num_basis_coeff_quadrilateral;
        get_quadrature_order_pointer =
            QuadrilateralQuadrature::get_gausslegendre_quadrature_order;
        get_quadrature_data =
            QuadrilateralQuadrature::get_quadrature_gauss_legendre;
        get_points_on_face = get_points_on_face_quadrilateral;
        get_face_pts_order_wrt_orient0 = get_face_pts_order_wrt_orient0_quadrilateral;

        // set constants
        NDIMS = 2;
        NFACES = 4;
        NCORNERS = 4;
        NUM_ORIENT_PER_FACE = 2;
        name = "Quadrilateral";
    }
    if (shape_type == ShapeType::Hexahedron){
        // set methods
        get_num_basis_coeff = get_num_basis_coeff_hexahedron;
        get_quadrature_order_pointer =
            QuadratureTools::get_gausslegendre_quadrature_order;
        get_quadrature_data =
            HexahedronQuadrature::get_quadrature_gauss_legendre;
        
        // set constants
        NDIMS = 3;
        NCORNERS = 8;
        NFACES = 6;
        NUM_ORIENT_PER_FACE = 8;
        name = "Hexahedron";
    }
}

int Shape::get_quadrature_order(const int order_) {
    return get_quadrature_order_pointer(order_, NDIMS);
}

} // end namespace Basis
