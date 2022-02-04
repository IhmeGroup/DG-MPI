#include "numerics/basis/tools.h"

namespace BasisTools {

DG_KOKKOS_FUNCTION
void equidistant_nodes_1D_range(rtype start, rtype stop, int nnodes,
	Kokkos::View<rtype*> &xnodes) {

	if (nnodes <= 1){
		xnodes(0) = 0.;
		return;
	}
	if (stop <= start) {
		throw ValueErrorException("Assume beginning is smaller than end");
	}
	if (xnodes.extent(0) != nnodes){
		Kokkos::resize(xnodes, nnodes);
	}

	rtype dx = (stop - start) / ((rtype)nnodes - 1.);

	for (int i = 0; i < nnodes; i++){
		xnodes(i) = start + (rtype)i * dx;
	}
} 

DG_KOKKOS_FUNCTION
void get_lagrange_basis_val_1D(const rtype &x, 
	Kokkos::View<const rtype*> xnodes, int p, 
	Kokkos::View<rtype*> phi){

	int nnodes = xnodes.extent(0);
	for (int i = 0; i < nnodes; i++){
		phi(i) = 1.;
		for (int j = 0; j < nnodes; j++){
			if (i == j) continue;
			phi(i) *= (x - xnodes(j)) / (xnodes(i) - xnodes(j));
		}
	}
}

DG_KOKKOS_FUNCTION
void get_lagrange_basis_grad_1D(const rtype &x, 
	Kokkos::View<const rtype*> xnodes, int p, 
	Kokkos::View<rtype*> gphi){

	int nnodes = xnodes.extent(0);
	for (int i = 0; i < nnodes; i++) {
		gphi(i) = 0.;
		for (int j = 0; j < nnodes; j++) {
			if (i == j) continue;
			rtype prod = 1.;
			for (int k = 0; k < nnodes; k++) {
				if (i == k) continue;
				else if (j == k) prod *= 1. / (xnodes(i) - xnodes(k));
				else prod *= (x - xnodes(k)) / (xnodes(i) - xnodes(k));
			}
			gphi(i) += prod;
		}
	}

}

DG_KOKKOS_FUNCTION
void get_lagrange_basis_val_2D(Kokkos::View<const rtype**> quad_pts, 
	Kokkos::View<const rtype*> xnodes,
	int p, Kokkos::View<rtype**> basis_val) {

	// get shape of basis_val
	const int nq = quad_pts.extent(0);
	const int nb = basis_val.extent(1);

	// allocate the x and y basis vals
	Kokkos::View<rtype**> valx("basis_val_x", nq, p + 1);
	Kokkos::View<rtype**> valy("basis_val_y", nq, p + 1);

	for (int iq = 0; iq < nq; iq++){
		get_lagrange_basis_val_1D(quad_pts(iq, 0), xnodes, p, 
				Kokkos::subview(valx, iq, Kokkos::ALL()));
		get_lagrange_basis_val_1D(quad_pts(iq, 1), xnodes, p,
				Kokkos::subview(valy, iq, Kokkos::ALL()));
	}
	for (int iq = 0; iq < nq; iq++){
		for (int j = 0; j < p + 1; j++){
			for (int i = 0; i < p + 1; i++){
				basis_val(iq, j * (p + 1) + i) = valx(iq, i) * valy(iq, j);
			}
		}
	}
}


DG_KOKKOS_FUNCTION
void get_lagrange_basis_grad_2D(Kokkos::View<const rtype**> quad_pts, 
	Kokkos::View<const rtype*> xnodes,
	int p, Kokkos::View<rtype***> basis_ref_grad) {
	// get shape of basis_ref_grad
	const int nq = quad_pts.extent(0);
	const int nb = basis_ref_grad.extent(1);
	const int ndims = basis_ref_grad.extent(2);

	// allocate the x and y basis vals + basis gradients
	Kokkos::View<rtype**> valx("basis_val_x", nq, p + 1);
	Kokkos::View<rtype**> valy("basis_val_y", nq, p + 1);
	Kokkos::View<rtype***> gradx("basis_grad_x", nq, p + 1, 1);
	Kokkos::View<rtype***> grady("basis_grad_y", nq, p + 1, 1);

	for (int iq = 0; iq < nq; iq++){
		get_lagrange_basis_val_1D(quad_pts(iq, 0), xnodes, p, 
				Kokkos::subview(valx, iq, Kokkos::ALL()));
		get_lagrange_basis_val_1D(quad_pts(iq, 1), xnodes, p,
				Kokkos::subview(valy, iq, Kokkos::ALL()));
		get_lagrange_basis_grad_1D(quad_pts(iq, 0), xnodes, p,
				Kokkos::subview(gradx, iq, Kokkos::ALL(), 0));
		get_lagrange_basis_grad_1D(quad_pts(iq, 1), xnodes, p,
				Kokkos::subview(grady, iq, Kokkos::ALL(), 0));		
	}

	for (int iq = 0; iq < nq; iq++){
		for (int j = 0; j < p + 1; j++){
			for (int i = 0; i < p + 1; i++){
				basis_ref_grad(iq, j * (p + 1) + i, 0) = gradx(iq, i, 0) * valy(iq, j);
				basis_ref_grad(iq, j * (p + 1) + i, 1) = valx(iq, i) * grady(iq, j, 0);
			}
		}
	}
}

DG_KOKKOS_FUNCTION
void get_lagrange_basis_val_3D(Kokkos::View<const rtype**> quad_pts, 
	Kokkos::View<const rtype*> xnodes,
	int p, Kokkos::View<rtype**> basis_val) {

	// get shape of basis_val
	const int nq = quad_pts.extent(0);
	const int nb = basis_val.extent(1);

	// allocate the x and y basis vals
	Kokkos::View<rtype**> valx("basis_val_x", nq, p + 1);
	Kokkos::View<rtype**> valy("basis_val_y", nq, p + 1);
	Kokkos::View<rtype**> valz("basis_val_z", nq, p + 1);

	for (int iq = 0; iq < nq; iq++){
		get_lagrange_basis_val_1D(quad_pts(iq, 0), xnodes, p, 
				Kokkos::subview(valx, iq, Kokkos::ALL()));
		get_lagrange_basis_val_1D(quad_pts(iq, 1), xnodes, p,
				Kokkos::subview(valy, iq, Kokkos::ALL()));
		get_lagrange_basis_val_1D(quad_pts(iq, 2), xnodes, p,
				Kokkos::subview(valz, iq, Kokkos::ALL()));
	}
	for (int iq = 0; iq < nq; iq++){
		for (int k = 0; k < p + 1; k++){
			for (int j = 0; j < p + 1; j++){
				for (int i = 0; i < p + 1; i++){
					basis_val(iq, k * (p + 1) * (p + 1) + j * (p + 1) + i) =
							valx(iq, i) * valy(iq, j) * valz(iq, k);
				}
			}
		}
	}
}


DG_KOKKOS_FUNCTION
void get_lagrange_basis_grad_3D(Kokkos::View<const rtype**> quad_pts, 
	Kokkos::View<const rtype*> xnodes,
	int p, Kokkos::View<rtype***> basis_ref_grad) {
	// get shape of basis_ref_grad
	const int nq = quad_pts.extent(0);
	const int nb = basis_ref_grad.extent(1);
	const int ndims = basis_ref_grad.extent(2);

	// // allocate the x and y basis vals + basis gradients
	Kokkos::View<rtype**> valx("basis_val_x", nq, p + 1);
	Kokkos::View<rtype**> valy("basis_val_y", nq, p + 1);
	Kokkos::View<rtype**> valz("basis_val_z", nq, p + 1);
	Kokkos::View<rtype***> gradx("basis_grad_x", nq, p + 1, 1);
	Kokkos::View<rtype***> grady("basis_grad_y", nq, p + 1, 1);
	Kokkos::View<rtype***> gradz("basis_grad_z", nq, p + 1, 1);

	for (int iq = 0; iq < nq; iq++){
		get_lagrange_basis_val_1D(quad_pts(iq, 0), xnodes, p, 
				Kokkos::subview(valx, iq, Kokkos::ALL()));
		get_lagrange_basis_val_1D(quad_pts(iq, 1), xnodes, p,
				Kokkos::subview(valy, iq, Kokkos::ALL()));
		get_lagrange_basis_val_1D(quad_pts(iq, 2), xnodes, p,
				Kokkos::subview(valz, iq, Kokkos::ALL()));
		get_lagrange_basis_grad_1D(quad_pts(iq, 0), xnodes, p,
				Kokkos::subview(gradx, iq, Kokkos::ALL(), 0));
		get_lagrange_basis_grad_1D(quad_pts(iq, 1), xnodes, p,
				Kokkos::subview(grady, iq, Kokkos::ALL(), 0));
		get_lagrange_basis_grad_1D(quad_pts(iq, 2), xnodes, p,
				Kokkos::subview(gradz, iq, Kokkos::ALL(), 0));		
	}

	for (int iq = 0; iq < nq; iq++){
		for (int k = 0; k < p + 1; k++){
			for (int j = 0; j < p + 1; j++){
				for (int i = 0; i < p + 1; i++){
					basis_ref_grad(iq, k * (p + 1) * (p + 1) + j * (p + 1) + i, 0) = 
							gradx(iq, i, 0) * valy(iq, j) * valz(iq, k);
					basis_ref_grad(iq, k * (p + 1) * (p + 1) + j * (p + 1) + i, 1) = 
							valx(iq, i) * grady(iq, j, 0) * valz(iq, k);
					basis_ref_grad(iq, k * (p + 1) * (p + 1) + j * (p + 1) + i, 2) = 
							valx(iq, i) * valy(iq, j) * gradz(iq, k, 0);
				}
			}
		}
	}
}

DG_KOKKOS_FUNCTION
void get_legendre_basis_val_1D(const rtype &x, const int p, 
	Kokkos::View<rtype*> phi) {
    if(p >= 0) {
       phi[0] = 1.;
    }
    if(p >= 1){
        phi[1] = x;
    }
    if(p >= 2){
        phi[2] = 0.5*(3.*x*x - 1.);
    }
    if(p >= 3){
        phi[3] = 0.5*(5.*x*x*x - 3.*x);
    }
    if(p >= 4){
        phi[4] = 0.125*(35.*x*x*x*x - 30.*x*x + 3.);
    }
    if(p >= 5){
        phi[5] = 0.125*(63.*x*x*x*x*x - 70.*x*x*x + 15.*x);
    }
    if(p >= 6){
        phi[6] = 0.0625*(231.*x*x*x*x*x*x - 315.*x*x*x*x + 105.*x*x -5.);
    }
    if(p == 7){
        phi[7] = 0.0625*(429.*x*x*x*x*x*x*x - 693.*x*x*x*x*x + 315.*x*x*x - 35.*x);
    }
    if(p > 7){
        std::string error_message = "Legendre polynomial P = ";
        error_message += std::to_string(p);
        error_message += " not implemented.";
        throw std::runtime_error(error_message);
    }

}

DG_KOKKOS_FUNCTION
void get_legendre_basis_grad_1D(const rtype &x, const int p, 
	Kokkos::View<rtype*> gphi) {
    if(p >= 0) {
        gphi[0] = 0.0;
    }
    if(p >= 1){
        gphi[1] = 1.;
    }
    if(p >= 2){
        gphi[2] = 3.*x;
    }
    if(p >= 3){
        gphi[3] = 0.5*(15.*x*x - 3.);
    }
    if(p >= 4){
        gphi[4] = 0.125*(35.*4.*x*x*x - 60.*x);
    }
    if(p >= 5){
        gphi[5] = 0.125*(63.*5.*x*x*x*x - 210.*x*x + 15.);
    }
    if(p >= 6){
        gphi[6] = 0.0625*(231.*6.*x*x*x*x*x - 315.*4.*x*x*x + 210.*x);
    }
    if(p == 7){
        gphi[7] = 0.0625*(429.*7.*x*x*x*x*x*x - 693.*5.*x*x*x*x + 315.*3.*x*x - 35.);
    }
    if(p > 7){
        std::string error_message = "Legendre polynomial P = ";
        error_message += std::to_string(p);
        error_message += " not implemented.";
        throw std::runtime_error(error_message);
    }
}

DG_KOKKOS_FUNCTION
void get_legendre_basis_val_2D(Kokkos::View<const rtype**> quad_pts, 
		const int p, Kokkos::View<rtype**> basis_val){
	// get shape of basis_val
	const int nq = quad_pts.extent(0);
	const int nb = basis_val.extent(1);

	// allocate the x and y basis vals
	Kokkos::View<rtype**> valx("basis_val_x", nq, p + 1);
	Kokkos::View<rtype**> valy("basis_val_y", nq, p + 1);

	for (int iq = 0; iq < nq; iq++){
		get_legendre_basis_val_1D(quad_pts(iq, 0), p, 
				Kokkos::subview(valx, iq, Kokkos::ALL()));
		get_legendre_basis_val_1D(quad_pts(iq, 1), p,
				Kokkos::subview(valy, iq, Kokkos::ALL()));
	}
	for (int iq = 0; iq < nq; iq++){
		for (int j = 0; j < p + 1; j++){
			for (int i = 0; i < p + 1; i++){
				basis_val(iq, j * (p + 1) + i) = valx(iq, i) * valy(iq, j);
			}
		}
	}
}

DG_KOKKOS_FUNCTION
void get_legendre_basis_grad_2D(Kokkos::View<const rtype**> quad_pts,
		const int p, Kokkos::View<rtype***> basis_ref_grad){
	// get shape of basis_ref_grad
	const int nq = quad_pts.extent(0);
	const int nb = basis_ref_grad.extent(1);
	const int ndims = basis_ref_grad.extent(2);

	// allocate the x and y basis vals + basis gradients
	Kokkos::View<rtype**> valx("basis_val_x", nq, p + 1);
	Kokkos::View<rtype**> valy("basis_val_y", nq, p + 1);
	Kokkos::View<rtype***> gradx("basis_grad_x", nq, p + 1, 1);
	Kokkos::View<rtype***> grady("basis_grad_y", nq, p + 1, 1);

	for (int iq = 0; iq < nq; iq++){
		get_legendre_basis_val_1D(quad_pts(iq, 0), p, 
				Kokkos::subview(valx, iq, Kokkos::ALL()));
		get_legendre_basis_val_1D(quad_pts(iq, 1), p,
				Kokkos::subview(valy, iq, Kokkos::ALL()));
		get_legendre_basis_grad_1D(quad_pts(iq, 0), p,
				Kokkos::subview(gradx, iq, Kokkos::ALL(), 0));
		get_legendre_basis_grad_1D(quad_pts(iq, 1), p,
				Kokkos::subview(grady, iq, Kokkos::ALL(), 0));		
	}

	for (int iq = 0; iq < nq; iq++){
		for (int j = 0; j < p + 1; j++){
			for (int i = 0; i < p + 1; i++){
				basis_ref_grad(iq, j * (p + 1) + i, 0) = gradx(iq, i, 0) * valy(iq, j);
				basis_ref_grad(iq, j * (p + 1) + i, 1) = valx(iq, i) * grady(iq, j, 0);
			}
		}
	}
}

DG_KOKKOS_FUNCTION
void get_legendre_basis_val_3D(Kokkos::View<const rtype**> quad_pts,
	const int p, Kokkos::View<rtype**> basis_val){
	// get shape of basis_val
	const int nq = quad_pts.extent(0);
	const int nb = basis_val.extent(1);

	// allocate the x and y basis vals
	Kokkos::View<rtype**> valx("basis_val_x", nq, p + 1);
	Kokkos::View<rtype**> valy("basis_val_y", nq, p + 1);
	Kokkos::View<rtype**> valz("basis_val_z", nq, p + 1);

	for (int iq = 0; iq < nq; iq++){
		get_legendre_basis_val_1D(quad_pts(iq, 0), p, 
				Kokkos::subview(valx, iq, Kokkos::ALL()));
		get_legendre_basis_val_1D(quad_pts(iq, 1), p,
				Kokkos::subview(valy, iq, Kokkos::ALL()));
		get_legendre_basis_val_1D(quad_pts(iq, 2), p,
				Kokkos::subview(valz, iq, Kokkos::ALL()));
	}
	for (int iq = 0; iq < nq; iq++){
		for (int k = 0; k < p + 1; k++){
			for (int j = 0; j < p + 1; j++){
				for (int i = 0; i < p + 1; i++){
					basis_val(iq, k * (p + 1) * (p + 1) + j * (p + 1) + i) =
							valx(iq, i) * valy(iq, j) * valz(iq, k);
				}
			}
		}
	}

}

DG_KOKKOS_FUNCTION
void get_legendre_basis_grad_3D(Kokkos::View<const rtype**> quad_pts,
	const int p, Kokkos::View<rtype***> basis_ref_grad){
	// get shape of basis_ref_grad
	const int nq = quad_pts.extent(0);
	const int nb = basis_ref_grad.extent(1);
	const int ndims = basis_ref_grad.extent(2);

	// allocate the x and y basis vals + basis gradients
	Kokkos::View<rtype**> valx("basis_val_x", nq, p + 1);
	Kokkos::View<rtype**> valy("basis_val_y", nq, p + 1);
	Kokkos::View<rtype**> valz("basis_val_z", nq, p + 1);
	Kokkos::View<rtype***> gradx("basis_grad_x", nq, p + 1, 1);
	Kokkos::View<rtype***> grady("basis_grad_y", nq, p + 1, 1);
	Kokkos::View<rtype***> gradz("basis_grad_z", nq, p + 1, 1);

	for (int iq = 0; iq < nq; iq++){
		get_legendre_basis_val_1D(quad_pts(iq, 0), p, 
				Kokkos::subview(valx, iq, Kokkos::ALL()));
		get_legendre_basis_val_1D(quad_pts(iq, 1), p,
				Kokkos::subview(valy, iq, Kokkos::ALL()));
		get_legendre_basis_val_1D(quad_pts(iq, 2), p,
				Kokkos::subview(valz, iq, Kokkos::ALL()));
		get_legendre_basis_grad_1D(quad_pts(iq, 0), p,
				Kokkos::subview(gradx, iq, Kokkos::ALL(), 0));
		get_legendre_basis_grad_1D(quad_pts(iq, 1), p,
				Kokkos::subview(grady, iq, Kokkos::ALL(), 0));	
		get_legendre_basis_grad_1D(quad_pts(iq, 2), p,
				Kokkos::subview(gradz, iq, Kokkos::ALL(), 0));		
	}

	for (int iq = 0; iq < nq; iq++){
		for (int k = 0; k < p + 1; k++){
			for (int j = 0; j < p + 1; j++){
				for (int i = 0; i < p + 1; i++){
					basis_ref_grad(iq, k * (p + 1) * (p + 1) + j * (p + 1) + i, 0) = 
							gradx(iq, i, 0) * valy(iq, j) * valz(iq, k);
					basis_ref_grad(iq, k * (p + 1) * (p + 1) + j * (p + 1) + i, 1) = 
							valx(iq, i) * grady(iq, j, 0) * valz(iq, k);
					basis_ref_grad(iq, k * (p + 1) * (p + 1) + j * (p + 1) + i, 2) = 
							valx(iq, i) * valy(iq, j) * gradz(iq, k, 0);

				}
			}
		}
	}
}


} // end namespace BasisTools