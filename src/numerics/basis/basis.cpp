#include "numerics/basis/basis.h"

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


/* --------------------------------------
	LegendreSeg Method Definitions
----------------------------------------*/
LegendreSeg::LegendreSeg(const int order){

	nb = get_num_basis_coeff(order);
}

} // end namespace Basis