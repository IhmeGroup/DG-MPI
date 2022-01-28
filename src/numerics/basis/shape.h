#ifndef DG_NUMERICS_SHAPE_H
#define DG_NUMERICS_SHAPE_H

#include "common/defines.h"
#include "common/my_exceptions.h"

#include <Kokkos_Core.hpp>

namespace Basis {

/*
This is a Mixin class used to represent a shape. Supported shapes include
point, segment, quadrilateral, triangle, hexahedron, tetrahedron, and prism
*/
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

	
	// Gets local IDs of principal nodes on face

	// Inputs:
	// -------
	// 	p: order of polynomial space
	// 	face_ID: reference element face value

	// Outputs:
	// --------
	// 	fnode_nums: local IDs of principal nodes on face
	
	// void get_local_face_principal_node_nums(int p, int face_ID,
	// 	Kokkos::View<rtype*> fnode_nums);
};

class QuadShape : public ShapeBase {
public:
	/*
	Class constructor
	*/
	QuadShape();

	int get_num_basis_coeff(int p) override;
};

class HexShape : public ShapeBase {
public:
	/*
	Class constructor
	*/
	HexShape();

	int get_num_basis_coeff(int p) override;
};

} // end namespace Basis

#endif // DG_NUMERICS_SHAPE_H