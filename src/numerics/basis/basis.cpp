#include "numerics/basis/basis.h"

namespace Basis {


void BasisBase::get_values(Kokkos::View<const rtype**> quad_pts,
		Kokkos::View<rtype**> basis_val){
	throw NotImplementedException("BasisBase does not implement "
                        "get_values -> implement in child class");
}

void BasisBase::get_grads(Kokkos::View<const rtype**> quad_pts,
		Kokkos::View<rtype***> basis_ref_grad){
	throw NotImplementedException("BasisBase does not implement "
                        "get_grads -> implement in child class");
}



/* --------------------------------------
	LagrangeSeg Method Definitions
----------------------------------------*/
LagrangeSeg::LagrangeSeg(const int order){

	this->order = order;
	nb = get_num_basis_coeff(order);

	// Set equidistant or GLL nodes (currently only supporting EQ)
	get_1d_nodes = &BasisTools::equidistant_nodes_1D_range;
}

void LagrangeSeg::get_values(Kokkos::View<const rtype**> quad_pts,
		Kokkos::View<rtype**> basis_val){

	int nq = quad_pts.extent(0);

	if (order == 0){
		KokkosBlas::fill(basis_val, 1.);
	}
	else {
		Kokkos::View<rtype*> xnodes("xnodes", order + 1);
		get_1d_nodes(-1., 1., order + 1, xnodes);
   		for (int iq = 0; iq < nq; iq++){
        	BasisTools::get_lagrange_basis_val_1D(quad_pts(iq, 0), xnodes, order, 
        		Kokkos::subview(basis_val, iq, Kokkos::ALL()));
    	}
	}

}

void LagrangeSeg::get_grads(Kokkos::View<const rtype**> quad_pts,
		Kokkos::View<rtype***> basis_ref_grad){

	int nq = quad_pts.extent(0);

	if (order > 0){
		Kokkos::View<rtype*> xnodes("xnodes", order + 1);
		get_1d_nodes(-1., 1., order + 1, xnodes);
		for (int iq = 0; iq < nq; iq++){
			BasisTools::get_lagrange_basis_grad_1D(quad_pts(iq, 0), xnodes, order,
				Kokkos::subview(basis_ref_grad, iq, Kokkos::ALL(), 0));
		}
	}
}


/* --------------------------------------
	LagrangeQuad Method Definitions
----------------------------------------*/
LagrangeQuad::LagrangeQuad(const int order){

	this->order = order;
	nb = get_num_basis_coeff(order);

	// Set equidistant or GLL nodes (currently only supporting EQ)
	get_1d_nodes = &BasisTools::equidistant_nodes_1D_range;
}

void LagrangeQuad::get_values(Kokkos::View<const rtype**> quad_pts,
		Kokkos::View<rtype**> basis_val){

	int nq = quad_pts.extent(0);

	if (order == 0){
		KokkosBlas::fill(basis_val, 1.);
	}
	else {
		Kokkos::View<rtype*> xnodes("xnodes", order + 1);
		get_1d_nodes(-1., 1., order + 1, xnodes);
   		for (int iq = 0; iq < nq; iq++){
        	BasisTools::get_lagrange_basis_val_2D(quad_pts, xnodes, order, 
        		basis_val);
    	}
	}

}

void LagrangeQuad::get_grads(Kokkos::View<const rtype**> quad_pts,
		Kokkos::View<rtype***> basis_ref_grad){

	int nq = quad_pts.extent(0);

	if (order > 0){
		Kokkos::View<rtype*> xnodes("xnodes", order + 1);
		get_1d_nodes(-1., 1., order + 1, xnodes);
		for (int iq = 0; iq < nq; iq++){
			BasisTools::get_lagrange_basis_grad_2D(quad_pts, xnodes, order,
				basis_ref_grad);
		}
	}
}

/* --------------------------------------
	LagrangeHex Method Definitions
----------------------------------------*/
LagrangeHex::LagrangeHex(const int order){

	this->order = order;
	nb = get_num_basis_coeff(order);

	// Set equidistant or GLL nodes (currently only supporting EQ)
	get_1d_nodes = &BasisTools::equidistant_nodes_1D_range;
}

void LagrangeHex::get_values(Kokkos::View<const rtype**> quad_pts,
		Kokkos::View<rtype**> basis_val){

	int nq = quad_pts.extent(0);

	if (order == 0){
		KokkosBlas::fill(basis_val, 1.);
	}
	else {
		Kokkos::View<rtype*> xnodes("xnodes", order + 1);
		get_1d_nodes(-1., 1., order + 1, xnodes);
   		for (int iq = 0; iq < nq; iq++){
        	BasisTools::get_lagrange_basis_val_3D(quad_pts, xnodes, order, 
        		basis_val);
    	}
	}

}

void LagrangeHex::get_grads(Kokkos::View<const rtype**> quad_pts,
		Kokkos::View<rtype***> basis_ref_grad){

	int nq = quad_pts.extent(0);

	if (order > 0){
		Kokkos::View<rtype*> xnodes("xnodes", order + 1);
		get_1d_nodes(-1., 1., order + 1, xnodes);
		for (int iq = 0; iq < nq; iq++){
			BasisTools::get_lagrange_basis_grad_3D(quad_pts, xnodes, order,
				basis_ref_grad);
		}
	}
}


/* --------------------------------------
	LegendreSeg Method Definitions
----------------------------------------*/
LegendreSeg::LegendreSeg(const int order){

	this->order = order;
	nb = get_num_basis_coeff(order);
}

void LegendreSeg::get_values(Kokkos::View<const rtype**> quad_pts,
		Kokkos::View<rtype**> basis_val){

	int nq = quad_pts.extent(0);

	if (order == 0){
		KokkosBlas::fill(basis_val, 1.);
	}
	else {
   		for (int iq = 0; iq < nq; iq++){
        	BasisTools::get_legendre_basis_val_1D(quad_pts(iq, 0), order, 
        		Kokkos::subview(basis_val, iq, Kokkos::ALL()));
    	}
	}

}

void LegendreSeg::get_grads(Kokkos::View<const rtype**> quad_pts,
		Kokkos::View<rtype***> basis_ref_grad){

	int nq = quad_pts.extent(0);

	if (order > 0){
		for (int iq = 0; iq < nq; iq++){
			BasisTools::get_legendre_basis_grad_1D(quad_pts(iq, 0), order,
				Kokkos::subview(basis_ref_grad, iq, Kokkos::ALL(), 0));
		}
	}
}



/* --------------------------------------
	LegendreQuad Method Definitions
----------------------------------------*/
LegendreQuad::LegendreQuad(const int order){

	this->order = order;
	nb = get_num_basis_coeff(order);
}

void LegendreQuad::get_values(Kokkos::View<const rtype**> quad_pts,
		Kokkos::View<rtype**> basis_val){

	int nq = quad_pts.extent(0);

	if (order == 0){
		KokkosBlas::fill(basis_val, 1.);
	}
	else {
    	BasisTools::get_legendre_basis_val_2D(quad_pts, order, 
    		basis_val);
	}
}


void LegendreQuad::get_grads(Kokkos::View<const rtype**> quad_pts,
		Kokkos::View<rtype***> basis_ref_grad){

	int nq = quad_pts.extent(0);

	if (order > 0){
		BasisTools::get_legendre_basis_grad_2D(quad_pts, order,
			basis_ref_grad);	
	}
}

/* --------------------------------------
	LegendreHex Method Definitions
----------------------------------------*/
LegendreHex::LegendreHex(const int order){

	this->order = order;
	nb = get_num_basis_coeff(order);
}

void LegendreHex::get_values(Kokkos::View<const rtype**> quad_pts,
		Kokkos::View<rtype**> basis_val){

	int nq = quad_pts.extent(0);

	if (order == 0){
		KokkosBlas::fill(basis_val, 1.);
	}
	else {
    	BasisTools::get_legendre_basis_val_3D(quad_pts, order, 
    		basis_val);
	}
}


void LegendreHex::get_grads(Kokkos::View<const rtype**> quad_pts,
		Kokkos::View<rtype***> basis_ref_grad){

	int nq = quad_pts.extent(0);

	if (order > 0){
		BasisTools::get_legendre_basis_grad_3D(quad_pts, order,
			basis_ref_grad);	
	}
}




} // end namespace Basis