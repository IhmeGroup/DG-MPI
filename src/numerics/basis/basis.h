#ifndef DG_NUMERICS_BASIS_H
#define DG_NUMERICS_BASIS_H

#include "common/defines.h"
#include "common/my_exceptions.h"
#include "numerics/basis/shape.h"
#include "numerics/basis/tools.h"

#include "common/enums.h"

#include <Kokkos_Core.hpp>
#include <KokkosBlas1_fill.hpp>

namespace Basis {


void get_values_lagrangeseg(host_view_type_2D quad_pts,
		host_view_type_2D basis_val, const int order, 
		void (*get_1d_nodes)(rtype, rtype, int,
		host_view_type_1D &));

void get_grads_lagrangeseg(host_view_type_2D quad_pts,
		host_view_type_3D basis_ref_grad, const int order, 
		void (*get_1d_nodes)(rtype, rtype, int,
		host_view_type_1D &));

void get_values_lagrangequad(host_view_type_2D quad_pts,
		host_view_type_2D basis_val, const int order, 
		void (*get_1d_nodes)(rtype start, rtype stop, int nnodes,
		host_view_type_1D &xnodes));

void get_grads_lagrangequad(host_view_type_2D quad_pts,
		host_view_type_3D basis_ref_grad, const int order, 
		void (*get_1d_nodes)(rtype start, rtype stop, int nnodes,
		host_view_type_1D &xnodes));

void get_values_lagrangehex(host_view_type_2D quad_pts,
		host_view_type_2D basis_val, const int order, 
		void (*get_1d_nodes)(rtype start, rtype stop, int nnodes,
		host_view_type_1D &xnodes));

void get_grads_lagrangehex(host_view_type_2D quad_pts,
		host_view_type_3D basis_ref_grad, const int order, 
		void (*get_1d_nodes)(rtype start, rtype stop, int nnodes,
		host_view_type_1D &xnodes));


void get_values_legendreseg(host_view_type_2D quad_pts,
		host_view_type_2D basis_val, const int order, 
		void (*get_1d_nodes)(rtype, rtype, int,
		host_view_type_1D &));

void get_grads_legendreseg(host_view_type_2D quad_pts,
		host_view_type_3D basis_ref_grad, const int order, 
		void (*get_1d_nodes)(rtype, rtype, int,
		host_view_type_1D &));

void get_values_legendrequad(host_view_type_2D quad_pts,
		host_view_type_2D basis_val, const int order, 
		void (*get_1d_nodes)(rtype start, rtype stop, int nnodes,
		host_view_type_1D &xnodes));

void get_grads_legendrequad(host_view_type_2D quad_pts,
		host_view_type_3D basis_ref_grad, const int order, 
		void (*get_1d_nodes)(rtype start, rtype stop, int nnodes,
		host_view_type_1D &xnodes));

void get_values_legendrehex(host_view_type_2D quad_pts,
		host_view_type_2D basis_val, const int order, 
		void (*get_1d_nodes)(rtype start, rtype stop, int nnodes,
		host_view_type_1D &xnodes));

void get_grads_legendrehex(host_view_type_2D quad_pts,
		host_view_type_3D basis_ref_grad, const int order, 
		void (*get_1d_nodes)(rtype start, rtype stop, int nnodes,
		host_view_type_1D &xnodes));

class Basis {

public:

	/*
	Constructor
	*/
	Basis(BasisType basis_type, const int order);
	Basis() = default;
	~Basis() = default;

	inline int get_order(){return order;}
	inline std::string get_name(){return name;}
	inline int get_num_basis_coeffs(){return nb;}

	void (*get_1d_nodes)(rtype start, rtype stop, int nnodes,
		host_view_type_1D &xnodes);

	void get_values(host_view_type_2D quad_pts,
		host_view_type_2D basis_val);

	void get_grads(host_view_type_2D quad_pts,
		host_view_type_3D basis_ref_grad);

private:

	void (*get_values_pointer)(host_view_type_2D quad_pts,
		host_view_type_2D basis_val, const int order, 
		void (*get_1d_nodes)(rtype, rtype, int,
		host_view_type_1D &));

	void (*get_grads_pointer)(host_view_type_2D quad_pts,
		host_view_type_3D basis_ref_grad, const int order, 
		void (*get_1d_nodes)(rtype, rtype, int,
		host_view_type_1D &));

public:
	Shape shape;
protected:
	std::string name; // name of basis
	int nb; //number of polynomial coefficients
	int order; // polynomial or geometric order
};

} // end namespace Basis

#endif //DG_NUMERICS_BASIS_H