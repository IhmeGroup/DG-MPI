#include "numerics/basis/shape.h"

namespace Basis {


int get_num_basis_coeff_segment(int p){
    return p + 1;
}

int get_num_basis_coeff_quadrilateral(int p){
    return (p + 1) * (p + 1);
}

int get_num_basis_coeff_hexahedron(int p){
    return (p + 1) * (p + 1) * (p + 1);
}

Shape::Shape(ShapeType shape_type){
    if (shape_type == ShapeType::Segment){
        get_num_basis_coeff = get_num_basis_coeff_segment;
        get_quadrature_order_pointer =
            QuadratureTools::get_gausslegendre_quadrature_order;
        get_quadrature_data =
            SegmentQuadrature::get_quadrature_gauss_legendre;
        NDIMS = 1;
        name = "Segment";
    }
    if (shape_type == ShapeType::Quadrilateral){
        get_num_basis_coeff = get_num_basis_coeff_quadrilateral;
        get_quadrature_order_pointer =
            QuadratureTools::get_gausslegendre_quadrature_order;
        get_quadrature_data =
            QuadrilateralQuadrature::get_quadrature_gauss_legendre;
        NDIMS = 2;
        name = "Quadrilateral";
    }
    if (shape_type == ShapeType::Hexahedron){
        get_num_basis_coeff = get_num_basis_coeff_hexahedron;
        get_quadrature_order_pointer =
            QuadratureTools::get_gausslegendre_quadrature_order;
        get_quadrature_data =
            HexahedronQuadrature::get_quadrature_gauss_legendre;
        NDIMS = 3;
        name = "Hexahedron";
    }
}

int Shape::get_quadrature_order(const int order_) {
    return get_quadrature_order_pointer(order_, NDIMS);
}

int ShapeBase::get_num_basis_coeff(int p){
    throw NotImplementedException("ShapeBase does not implement "
                                      "get_num_basis_coeff -> implement in child class");
}

} // end namespace Basis
