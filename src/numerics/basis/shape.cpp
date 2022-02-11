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

// /* --------------------------------------
// 	PointShape Method Definitions
// ----------------------------------------*/
// PointShape::PointShape(){
// 	NFACES = 0;
// 	NDIMS = 0;
// }

// int PointShape::get_num_basis_coeff(int p){
// 	return 1;
// }


// /* --------------------------------------
// 	SegShape Method Definitions
// ----------------------------------------*/
// SegShape::SegShape(){
// 	NFACES = 2;
// 	NDIMS = 1;
// }

// int SegShape::get_num_basis_coeff(int p){
// 	return p + 1;
// }

// // void SegShape::get_local_face_principal_node_nums(int p, int face_ID,
// // 	Kokkos::View<rtype*> fnode_nums){}

// /* --------------------------------------
// 	QuadShape Method Definitions
// ----------------------------------------*/
// QuadShape::QuadShape(){
// 	NFACES = 4;
// 	NDIMS = 2;
// }

// int QuadShape::get_num_basis_coeff(int p){
// 	return (p + 1) * (p + 1);
// }


// /* --------------------------------------
// 	HexShape Method Definitions
// ----------------------------------------*/
// HexShape::HexShape(){
// 	NFACES = 6;
// 	NDIMS = 3;
// }

// int HexShape::get_num_basis_coeff(int p){
// 	return (p + 1) * (p + 1) * (p + 1);
// }

} // end namespace Basis