#ifndef DG_NUMERICS_BASIS_H
#define DG_NUMERICS_BASIS_H

#include "common/defines.h"
#include "common/my_exceptions.h"
#include "numerics/basis/shape.h"
#include "numerics/basis/tools.h"

#include <Kokkos_Core.hpp>
#include <KokkosBlas1_fill.hpp>

namespace Basis {


/*
This is a base class used for the base methods
of all available basis functions.

Child classes (available basis functions) of this base class include:
	- Lagrange basis [support for segments, quadrilaterals,
	  triangles, hexahedrons, and (eventually) tetrahedrons)]
	- Legendre basis [support for segments and quadrilaterals]
*/
class BasisBase {
public:
	/*
	Virtual destructor
	*/
	virtual ~BasisBase() = default;

	
	inline int get_order(){return order;}
	/*
	This is a function pointer that get assigned a function from
	the BasisTools namespace that either defines equidistant
	nodes or gauss-lobatto nodes. 

	NOTE: Currently only supporting
	equidistant nodes. This function is set in the constructor
	of the Lagrange classes for now but needs to be moved to a 
	setter function later once enums are figured out.

	This function gets the 1D coordinates in reference space

	Inputs:
	-------
		start: start of ref space (typically -1)
		stop:  end of ref space (typically 1)
		nnodes: num of nodes in 1D ref space

	Outputs:
	--------
		xnodes: coordinates of nodes in 1D ref space [nnodes]
	*/
	void (*get_1d_nodes)(rtype start, rtype stop, int nnodes,
		Kokkos::View<rtype*> &xnodes);

	/*
	Calculates the basis values

	Inputs:
	-------
		quad_pts: coordinates of quadrature points [nb, ndims]

	Outputs:
	--------
		basis_val: evaluated basis function [nq, nb]
	*/
	virtual void get_values(Kokkos::View<const rtype**> quad_pts,
		Kokkos::View<rtype**> basis_val);

	/*
	Calculates basis gradient (in reference space)

	Inputs:
	-------
		quad_pts: coordinates of quadrature points [nb, ndims]

	Outputs:
	--------
		basis_ref_grad: evaluated gradient of basis function in
			reference space [nq, nb, ndims]
	*/
	virtual void get_grads(Kokkos::View<const rtype**> quad_pts,
		Kokkos::View<rtype***> basis_ref_grad);

protected:
	int nb; //number of polynomial coefficients
	int order; // polynomial or geometric order
};


class LagrangeSeg : public BasisBase, public SegShape {
public:
	/*
	Class constructor
	*/
	LagrangeSeg(const int order);

	/*
	Class destructor
	*/
	~LagrangeSeg() = default;

	void get_values(Kokkos::View<const rtype**> quad_pts,
		Kokkos::View<rtype**> basis_val) override;

	void get_grads(Kokkos::View<const rtype**> quad_pts,
		Kokkos::View<rtype***> basis_ref_grad) override;
};

class LagrangeQuad : public BasisBase, public QuadShape {
public:
	/*
	Class constructor
	*/
	LagrangeQuad(const int order);

	/*
	Class destructor
	*/
	~LagrangeQuad() = default;

	void get_values(Kokkos::View<const rtype**> quad_pts,
		Kokkos::View<rtype**> basis_val) override;

	void get_grads(Kokkos::View<const rtype**> quad_pts,
		Kokkos::View<rtype***> basis_ref_grad) override;
};

class LagrangeHex : public BasisBase, public HexShape {
public:
	/*
	Class constructor
	*/
	LagrangeHex(const int order);

	/*
	Class destructor
	*/
	~LagrangeHex() = default;

	void get_values(Kokkos::View<const rtype**> quad_pts,
		Kokkos::View<rtype**> basis_val) override;

	void get_grads(Kokkos::View<const rtype**> quad_pts,
		Kokkos::View<rtype***> basis_ref_grad) override;
};


class LegendreSeg : public BasisBase, public SegShape {
public:
	/*
	Class constructor
	*/
	LegendreSeg(const int order);

	/*
	Class destructor
	*/
	~LegendreSeg() = default;

	void get_values(Kokkos::View<const rtype**> quad_pts,
		Kokkos::View<rtype**> basis_val) override;

	void get_grads(Kokkos::View<const rtype**> quad_pts,
		Kokkos::View<rtype***> basis_ref_grad) override;
};

class LegendreQuad : public BasisBase, public QuadShape {
public:
	/*
	Class constructor
	*/
	LegendreQuad(const int order);

	/*
	Class destructor
	*/
	~LegendreQuad() = default;

	void get_values(Kokkos::View<const rtype**> quad_pts,
		Kokkos::View<rtype**> basis_val) override;

	void get_grads(Kokkos::View<const rtype**> quad_pts,
		Kokkos::View<rtype***> basis_ref_grad) override;

};

class LegendreHex : public BasisBase, public HexShape {
public:
	/*
	Class constructor
	*/
	LegendreHex(const int order);

	/*
	Class destructor
	*/
	~LegendreHex() = default;

	void get_values(Kokkos::View<const rtype**> quad_pts,
		Kokkos::View<rtype**> basis_val) override;

	void get_grads(Kokkos::View<const rtype**> quad_pts,
		Kokkos::View<rtype***> basis_ref_grad) override;

};




} // end namespace Basis

#endif //DG_NUMERICS_BASIS_H