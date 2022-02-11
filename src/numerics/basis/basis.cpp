#include "numerics/basis/basis.h"

namespace Basis {

/* --------------------------------------
	LagrangeSeg Method Definitions
----------------------------------------*/
void get_values_lagrangeseg(host_view_type_2D quad_pts,
		host_view_type_2D basis_val, const int order, 
		void (*get_1d_nodes)(rtype, rtype, int,
		host_view_type_1D &)){

	int nq = quad_pts.extent(0);

	if (order == 0){
		KokkosBlas::fill(basis_val, 1.);
	}
	else {
		host_view_type_1D xnodes("xnodes", order + 1);
		get_1d_nodes(-1., 1., order + 1, xnodes);

   		for (int iq = 0; iq < nq; iq++){
        	BasisTools::get_lagrange_basis_val_1D(quad_pts(iq, 0), xnodes, order, 
        		Kokkos::subview(basis_val, iq, Kokkos::ALL()));
    	}
	}

}

void get_grads_lagrangeseg(host_view_type_2D quad_pts,
		host_view_type_3D basis_ref_grad, const int order, 
		void (*get_1d_nodes)(rtype start, rtype stop, int nnodes,
		host_view_type_1D &xnodes)){

	int nq = quad_pts.extent(0);

	if (order > 0){
		host_view_type_1D xnodes("xnodes", order + 1);
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
void get_values_lagrangequad(host_view_type_2D quad_pts,
		host_view_type_2D basis_val, const int order, 
		void (*get_1d_nodes)(rtype start, rtype stop, int nnodes,
		host_view_type_1D &xnodes)){

	int nq = quad_pts.extent(0);

	if (order == 0){
		KokkosBlas::fill(basis_val, 1.);
	}
	else {
		host_view_type_1D xnodes("xnodes", order + 1);
		get_1d_nodes(-1., 1., order + 1, xnodes);
   		for (int iq = 0; iq < nq; iq++){
        	BasisTools::get_lagrange_basis_val_2D(quad_pts, xnodes, order, 
        		basis_val);
    	}
	}

}

void get_grads_lagrangequad(host_view_type_2D quad_pts,
		host_view_type_3D basis_ref_grad, const int order, 
		void (*get_1d_nodes)(rtype start, rtype stop, int nnodes,
		host_view_type_1D &xnodes)){

	int nq = quad_pts.extent(0);

	if (order > 0){
		host_view_type_1D xnodes("xnodes", order + 1);
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
void get_values_lagrangehex(host_view_type_2D quad_pts,
		host_view_type_2D basis_val, const int order, 
		void (*get_1d_nodes)(rtype start, rtype stop, int nnodes,
		host_view_type_1D &xnodes)){

	int nq = quad_pts.extent(0);

	if (order == 0){
		KokkosBlas::fill(basis_val, 1.);
	}
	else {
		host_view_type_1D xnodes("xnodes", order + 1);
		get_1d_nodes(-1., 1., order + 1, xnodes);
   		for (int iq = 0; iq < nq; iq++){
        	BasisTools::get_lagrange_basis_val_3D(quad_pts, xnodes, order, 
        		basis_val);
    	}
	}
}

void get_grads_lagrangehex(host_view_type_2D quad_pts,
		host_view_type_3D basis_ref_grad, const int order, 
		void (*get_1d_nodes)(rtype start, rtype stop, int nnodes,
		host_view_type_1D &xnodes)){

	int nq = quad_pts.extent(0);

	if (order > 0){
		host_view_type_1D xnodes("xnodes", order + 1);
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
void get_values_legendreseg(host_view_type_2D quad_pts,
		host_view_type_2D basis_val, const int order, 
		void (*get_1d_nodes)(rtype start, rtype stop, int nnodes,
		host_view_type_1D &xnodes)){

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

void get_grads_legendreseg(host_view_type_2D quad_pts,
		host_view_type_3D basis_ref_grad, const int order, 
		void (*get_1d_nodes)(rtype start, rtype stop, int nnodes,
		host_view_type_1D &xnodes)){

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
void get_values_legendrequad(host_view_type_2D quad_pts,
		host_view_type_2D basis_val, const int order, 
		void (*get_1d_nodes)(rtype start, rtype stop, int nnodes,
		host_view_type_1D &xnodes)){

	int nq = quad_pts.extent(0);

	if (order == 0){
		KokkosBlas::fill(basis_val, 1.);
	}
	else {
    	BasisTools::get_legendre_basis_val_2D(quad_pts, order, 
    		basis_val);
	}
}


void get_grads_legendrequad(host_view_type_2D quad_pts,
		host_view_type_3D basis_ref_grad, const int order, 
		void (*get_1d_nodes)(rtype start, rtype stop, int nnodes,
		host_view_type_1D &xnodes)){

	int nq = quad_pts.extent(0);

	if (order > 0){
		BasisTools::get_legendre_basis_grad_2D(quad_pts, order,
			basis_ref_grad);	
	}
}

/* --------------------------------------
	LegendreHex Method Definitions
----------------------------------------*/
void get_values_legendrehex(host_view_type_2D quad_pts,
		host_view_type_2D basis_val, const int order, 
		void (*get_1d_nodes)(rtype start, rtype stop, int nnodes,
		host_view_type_1D &xnodes)){

	int nq = quad_pts.extent(0);

	if (order == 0){
		KokkosBlas::fill(basis_val, 1.);
	}
	else {
    	BasisTools::get_legendre_basis_val_3D(quad_pts, order, 
    		basis_val);
	}
}


void get_grads_legendrehex(host_view_type_2D quad_pts,
		host_view_type_3D basis_ref_grad, const int order, 
		void (*get_1d_nodes)(rtype start, rtype stop, int nnodes,
		host_view_type_1D &xnodes)){

	int nq = quad_pts.extent(0);

	if (order > 0){
		BasisTools::get_legendre_basis_grad_3D(quad_pts, order,
			basis_ref_grad);	
	}
}

/* ------------------------------------------------------
	Basis Method Definitions + Function Pointer Wrappers
--------------------------------------------------------*/
Basis::Basis(BasisType basis_type, const int order){

	this->order = order;
	get_1d_nodes = BasisTools::equidistant_nodes_1D_range;

	if (basis_type == BasisType::LagrangeSeg){
		get_values_pointer = get_values_lagrangeseg;
		get_grads_pointer = get_grads_lagrangeseg;
		name = "LagrangeSeg";
		shape = Shape(enum_from_string<ShapeType>("Segment"));
	}
	if (basis_type == BasisType::LagrangeQuad){
		get_values_pointer = get_values_lagrangequad;
		get_grads_pointer = get_grads_lagrangequad;
		name = "LagrangeQuad";
		shape = Shape(enum_from_string<ShapeType>("Quadrilateral"));
	}
	if (basis_type == BasisType::LagrangeHex){
		get_values_pointer = get_values_lagrangehex;
		get_grads_pointer = get_grads_lagrangehex;
		name = "LagrangeHex";
		shape = Shape(enum_from_string<ShapeType>("Hexahedron"));
	}
	if (basis_type == BasisType::LegendreSeg){
		get_values_pointer = get_values_legendreseg;
		get_grads_pointer = get_grads_legendreseg;
		name = "LegendreSeg";
		shape = Shape(enum_from_string<ShapeType>("Segment"));
	}
	if (basis_type == BasisType::LegendreQuad){
		get_values_pointer = get_values_legendrequad;
		get_grads_pointer = get_grads_legendrequad;
		name = "LegendreQuad";
		shape = Shape(enum_from_string<ShapeType>("Quadrilateral"));
	}
	if (basis_type == BasisType::LegendreHex){
		get_values_pointer = get_values_legendrehex;
		get_grads_pointer = get_grads_legendrehex;
		name = "LegendreHex";
		shape = Shape(enum_from_string<ShapeType>("Hexahedron"));
	}

}

void Basis::get_values(host_view_type_2D quad_pts, 
	host_view_type_2D basis_val) {

	get_values_pointer(quad_pts, basis_val, 
		order, get_1d_nodes);
}

void Basis::get_grads(host_view_type_2D quad_pts, 
	host_view_type_3D basis_ref_grad) {

	get_grads_pointer(quad_pts, basis_ref_grad, 
		order, get_1d_nodes);
}



} // end namespace Basis