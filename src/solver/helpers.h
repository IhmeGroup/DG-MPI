#ifndef DG_SOLVER_HELPERS_H
#define DG_SOLVER_HELPERS_H

#include "common/defines.h"
#include "mesh/mesh.h"
#include "mesh/tools.h"
#include "numerics/basis/basis.h"
#include "numerics/basis/tools.h"
#include "numerics/quadrature/tools.h"
#include <Kokkos_Core.hpp>
#include "KokkosBlas3_gemm.hpp"


namespace VolumeHelpers {

struct VolumeHelperFunctor {

    VolumeHelperFunctor() = default;

    inline
    void compute_volume_helpers(int scratch_size, Mesh& mesh,
        Basis::Basis& basis, MemoryNetwork& network);

    inline
    void compute_inv_mass_matrices(int scratch_size, Mesh& mesh,
        Basis::Basis& basis);

    inline
    void get_quadrature(Basis::Basis basis,
        const int order);

    inline
    void get_reference_data(Basis::Basis basis,
        Basis::Basis gbasis,
        const int order);

    inline
    void allocate_views(const int num_elems);

    view_type_2D quad_pts;
    view_type_1D quad_wts;

    host_view_type_2D h_quad_pts;
    host_view_type_1D h_quad_wts;

    view_type_2D basis_val;
    view_type_3D basis_ref_grad;

    host_view_type_2D h_basis_val;
    host_view_type_3D h_basis_ref_grad;

    view_type_2D gbasis_val;
    view_type_3D gbasis_ref_grad;

    host_view_type_2D h_gbasis_val;
    host_view_type_3D h_gbasis_ref_grad;

    view_type_4D jac_elems;
    view_type_2D djac_elems;
    view_type_4D ijac_elems;

    view_type_3D x_elems;
    view_type_1D vol_elems;

    view_type_3D iMM_elems;
};

template<typename ViewType2D, typename ViewType3D> inline
void evaluate_state(const int num_elems, ViewType2D basis_val, ViewType3D Uc, ViewType3D Uq);

} // end namespace VolumeHelper

namespace InteriorFaceHelpers {

struct InteriorFaceHelperFunctor{

    InteriorFaceHelperFunctor() = default;

    inline
    void compute_interior_face_helpers(int scratch_size, Mesh& mesh,
        Basis::Basis& basis);

    inline
    void get_quadrature(Basis::Basis basis,
        const int order);

    inline
    void get_reference_data(Basis::Basis basis, 
        Basis::Basis gbasis, const int order);

    inline
    void precompute_facequadrature_lookup(Mesh& mesh,
        Basis::Basis basis);

    view_type_3D quad_pts; // [NFACE, nq, NDIMS]
    view_type_1D quad_wts; // [nq]

    host_view_type_3D h_quad_pts; // [NFACE, nq, NDIMS]
    host_view_type_1D h_quad_wts; // [nq]

    view_type_3D basis_val; // [NFACE, nq, nb]
    view_type_4D basis_ref_grad; // [NFACE, nq, nb, NDIMS]

    host_view_type_3D h_basis_val; // [NFACE, nq, nb]
    host_view_type_4D h_basis_ref_grad; // [NFACE, nq, nb, NDIMS]

    view_type_3D gbasis_val; // [NFACE, nq, nb]
    view_type_4D gbasis_ref_grad; // [NFACE, nq, nb, NDIMS]

    host_view_type_3D h_gbasis_val; // [NFACE, nq, nb]
    host_view_type_4D h_gbasis_ref_grad; // [NFACE, nq, nb, NDIMS]

    Kokkos::View<int**> quad_idx_L; // [num_ifaces_part, nqf]
    Kokkos::View<int**> quad_idx_R; // [num_ifaces_part, nqf]

};

} // end interior face helper namespace
#include "solver/helpers.cpp"

#endif // DG_SOLVER_HELPERS_H
