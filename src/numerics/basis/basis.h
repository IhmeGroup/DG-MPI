#ifndef DG_NUMERICS_BASIS_H
#define DG_NUMERICS_BASIS_H

#include "common/defines.h"
#include "common/my_exceptions.h"
#include "numerics/basis/shape.h"
#include "numerics/basis/tools.h"

#include <Kokkos_Core.hpp>
#include <KokkosBlas1_fill.hpp>

namespace Basis {

class BasisBase {
public:
	/*
	Virtual destructor
	*/
	virtual ~BasisBase() = default;

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