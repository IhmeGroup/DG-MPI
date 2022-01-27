#ifndef DG_NUMERICS_BASIS_H
#define DG_NUMERICS_BASIS_H

#include "common/defines.h"
#include "common/my_exceptions.h"

namespace Basis {

class ShapeBase {
public:
	/*
	Virtual destructor
	*/
	virtual ~ShapeBase() = default;

	/*
	Sets the number of basis coefficients given a polynomial order

	Inputs:
	-------
		p: order of polynomial space

	Outputs:
	--------
		nb: number of basis coefficients
	*/
	virtual int get_num_basis_coeff(int p);

	inline int get_NFACES(){return NFACES;}

	inline int get_NDIMS(){return NDIMS;}

protected:
	int NDIMS; // number of dimensions
	int NFACES; // number of faces for shape type
};

class PointShape : public ShapeBase {
public:
	/*
	Class Constructor
	*/
	PointShape();
	
	int get_num_basis_coeff(int p) override;
};

class SegShape : public ShapeBase {
public:
	/*
	Class constructor
	*/
	SegShape();
	int get_num_basis_coeff(int p) override;
};






class BasisBase {
public:
	/*
	Virtual destructor
	*/
	virtual ~BasisBase() = default;

protected:
	int nb; //number of polynomial coefficients

};

class LegendreSeg : public BasisBase, public SegShape {
public:
	/*
	Class constructor
	*/
	LegendreSeg(const int order);


};


} // end namespace Basis

#endif //DG_NUMERICS_BASIS_H