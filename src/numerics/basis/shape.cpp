#include "numerics/basis/shape.h"



namespace Basis {

int ShapeBase::get_num_basis_coeff(int p){
	throw NotImplementedException("ShapeBase does not implement "
                                      "get_num_basis_coeff -> implement in child class");
}

/* --------------------------------------
	PointShape Method Definitions
----------------------------------------*/
PointShape::PointShape(){
	NFACES = 0;
	NDIMS = 0;
}

int PointShape::get_num_basis_coeff(int p){
	return 1;
}


/* --------------------------------------
	SegShape Method Definitions
----------------------------------------*/
SegShape::SegShape(){
	NFACES = 2;
	NDIMS = 1;
}

int SegShape::get_num_basis_coeff(int p){
	return p + 1;
}

// void SegShape::get_local_face_principal_node_nums(int p, int face_ID,
// 	Kokkos::View<rtype*> fnode_nums){}

/* --------------------------------------
	QuadShape Method Definitions
----------------------------------------*/
QuadShape::QuadShape(){
	NFACES = 4;
	NDIMS = 2;
}

int QuadShape::get_num_basis_coeff(int p){
	return (p + 1) * (p + 1);
}


/* --------------------------------------
	HexShape Method Definitions
----------------------------------------*/
HexShape::HexShape(){
	NFACES = 6;
	NDIMS = 3;
}

int HexShape::get_num_basis_coeff(int p){
	return (p + 1) * (p + 1) * (p + 1);
}

} // end namespace Basis