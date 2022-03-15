namespace VolumeHelpers {


inline
void VolumeHelperFunctor::compute_volume_helpers(int scratch_size,
    Mesh& mesh, Basis::Basis& basis){

    get_quadrature(basis, basis.get_order());
    get_reference_data(basis, mesh.gbasis, basis.get_order());
    allocate_views(mesh.num_elems_part);

    parallel_for("volume helpers", Kokkos::TeamPolicy<>( mesh.num_elems_part, 
        Kokkos::AUTO).set_scratch_size( 1, 
        Kokkos::PerTeam( scratch_size )), 
        KOKKOS_CLASS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& member) {

        const int elem_ID = member.league_rank();

        scratch_view_2D_rtype elem_coords(member.team_scratch( 1 ),
                mesh.num_nodes_per_elem, mesh.dim);

        if (member.team_rank() == 0 ) {
            MeshTools::elem_coords_from_elem_ID(mesh, elem_ID, elem_coords, member);

            // get the physical location of the quadrature points
            MeshTools::ref_to_phys(gbasis_val,
                Kokkos::subview(x_elems, elem_ID, Kokkos::ALL(), Kokkos::ALL()),
                elem_coords, member);

        }

        member.team_barrier();

        // get the determinant of the geometry jacobian and the inverse of
        // the jacobian for each element and store it on the device
        BasisTools::get_element_jacobian(quad_pts, gbasis_ref_grad,
            Kokkos::subview(jac_elems, elem_ID, Kokkos::ALL(),
            Kokkos::ALL(), Kokkos::ALL()),
            Kokkos::subview(djac_elems, elem_ID, Kokkos::ALL()),
            Kokkos::subview(ijac_elems, elem_ID, Kokkos::ALL(),
            Kokkos::ALL(), Kokkos::ALL()), elem_coords, member);

        member.team_barrier();

        // get the volume of each element
        MeshTools::get_element_volume(quad_wts,
            Kokkos::subview(djac_elems, elem_ID, Kokkos::ALL()),
            vol_elems(elem_ID), member);
    });

}

inline
void VolumeHelperFunctor::compute_inv_mass_matrices(int scratch_size, 
    Mesh& mesh, Basis::Basis& basis){

    // to correctly compute iMM we need 2*p for quadrature rules
    get_quadrature(basis, 2 * basis.get_order());
    get_reference_data(basis, mesh.gbasis, basis.get_order());

    const int nq = quad_pts.extent(0);
    const int nb = basis_val.extent(1);
    const int ndims = mesh.dim;
    const int num_elems = mesh.num_elems_part;

    Kokkos::resize(jac_elems, num_elems, nq, ndims, ndims);
    Kokkos::resize(djac_elems, num_elems, nq);
    Kokkos::resize(ijac_elems, num_elems, nq, ndims, ndims);
    Kokkos::resize(iMM_elems, num_elems, nb, nb);

    parallel_for("iMM_helper", Kokkos::TeamPolicy<>( mesh.num_elems_part, 
        Kokkos::AUTO).set_scratch_size( 1, 
        Kokkos::PerTeam( scratch_size )), 
        KOKKOS_CLASS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& member) {

                const int elem_ID = member.league_rank();

                scratch_view_2D_rtype elem_coords(member.team_scratch( 1 ),
                    mesh.num_nodes_per_elem, mesh.dim);

                if (member.team_rank() == 0 ) {
                    MeshTools::elem_coords_from_elem_ID(mesh, elem_ID, elem_coords, member);
                }
                member.team_barrier();

                // get the determinant of the geometry jacobian and the inverse of
                // the jacobian for each element and store it on the device
                BasisTools::get_element_jacobian(quad_pts, gbasis_ref_grad,
                    Kokkos::subview(jac_elems, elem_ID, Kokkos::ALL(),
                    Kokkos::ALL(), Kokkos::ALL()),
                    Kokkos::subview(djac_elems, elem_ID, Kokkos::ALL()),
                    Kokkos::subview(ijac_elems, elem_ID, Kokkos::ALL(),
                    Kokkos::ALL(), Kokkos::ALL()), elem_coords, member);

                BasisTools::get_inv_mass_matrices(quad_wts, basis_val,
                    Kokkos::subview(djac_elems, elem_ID, Kokkos::ALL()),
                    Kokkos::subview(iMM_elems, elem_ID, Kokkos::ALL(), Kokkos::ALL()), member);               

    });
}

inline
void VolumeHelperFunctor::allocate_views(const int num_elems){

    const int nq = quad_pts.extent(0);
    const int ndims = quad_pts.extent(1);

    Kokkos::resize(jac_elems, num_elems, nq, ndims, ndims);
    Kokkos::resize(djac_elems, num_elems, nq);
    Kokkos::resize(ijac_elems, num_elems, nq, ndims, ndims);
    Kokkos::resize(x_elems, num_elems, nq, ndims);
    Kokkos::resize(vol_elems, num_elems);
}


inline
void VolumeHelperFunctor::get_quadrature(
    Basis::Basis basis, const int order){

    // unpack
    int NDIMS = basis.shape.get_NDIMS();
    int nq_1d; int nq;
    int qorder = basis.shape.get_quadrature_order(order);
    QuadratureTools::get_number_of_quadrature_points(qorder, NDIMS,
            nq_1d, nq);

    // need to establish an initial size for views prior
    // to resizing inside of get_quadrature_data
    Kokkos::resize(quad_pts, nq, NDIMS);
    Kokkos::resize(quad_wts, nq);

    h_quad_pts = Kokkos::create_mirror_view(quad_pts);
    h_quad_wts = Kokkos::create_mirror_view(quad_wts);

    basis.shape.get_quadrature_data(qorder, nq_1d, h_quad_pts, h_quad_wts);

    Kokkos::deep_copy(quad_pts, h_quad_pts);
    Kokkos::deep_copy(quad_wts, h_quad_wts);

}


inline
void VolumeHelperFunctor::get_reference_data(
        Basis::Basis basis, Basis::Basis gbasis, const int order){
    // unpack
    int NDIMS = basis.shape.get_NDIMS();
    int nb = basis.shape.get_num_basis_coeff(order);
    int nq = quad_pts.extent(0);

    // need to establish an initial size for views
    Kokkos::resize(basis_val, nq, nb);
    Kokkos::resize(basis_ref_grad, nq, nb, NDIMS);

    h_basis_val = Kokkos::create_mirror_view(basis_val);
    h_basis_ref_grad = Kokkos::create_mirror_view(basis_ref_grad);

    basis.get_values(h_quad_pts, h_basis_val);
    basis.get_grads(h_quad_pts, h_basis_ref_grad);

    Kokkos::deep_copy(basis_val, h_basis_val);
    Kokkos::deep_copy(basis_ref_grad, h_basis_ref_grad);

    // repeat operations for the geometric basis
    // unpack
    int gorder = gbasis.get_order();
    NDIMS = gbasis.shape.get_NDIMS();
    nb = gbasis.shape.get_num_basis_coeff(gorder);

    // need to establish an initial size for views
    Kokkos::resize(gbasis_val, nq, nb);
    Kokkos::resize(gbasis_ref_grad, nq, nb, NDIMS);

    h_gbasis_val = Kokkos::create_mirror_view(gbasis_val);
    h_gbasis_ref_grad = Kokkos::create_mirror_view(gbasis_ref_grad);

    gbasis.get_values(h_quad_pts, h_gbasis_val);
    gbasis.get_grads(h_quad_pts, h_gbasis_ref_grad);

    Kokkos::deep_copy(gbasis_val, h_gbasis_val);
    Kokkos::deep_copy(gbasis_ref_grad, h_gbasis_ref_grad);

}

template<typename ViewType2D, typename ViewType3D> inline
void evaluate_state(const int num_elems, ViewType2D basis_val, ViewType3D Uc, ViewType3D Uq){
    
    Kokkos::parallel_for("eval state", Kokkos::TeamPolicy<>( num_elems,
        Kokkos::AUTO), KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& member){
        const int elem_ID = member.league_rank();
        auto Uq_elem = Kokkos::subview(Uq, elem_ID, Kokkos::ALL(), Kokkos::ALL());
        auto Uc_elem = Kokkos::subview(Uc, elem_ID, Kokkos::ALL(), Kokkos::ALL());
        
        Math::cAxB_to_C(1., basis_val, Uc_elem, Uq_elem, member);
        member.team_barrier();
    });

}
} // end namespace VolumeHelper



namespace InteriorFaceHelpers {

void InteriorFaceHelperFunctor::compute_interior_face_helpers(int scratch_size, Mesh& mesh,
    Basis::Basis& basis){

    get_quadrature(basis, basis.get_order());

    for (unsigned long i = 0; i < h_quad_pts.extent(0); i++){
        for (unsigned long j = 0; j < h_quad_pts.extent(1); j++){
        printf("qpts(%i, %i)=%f\n", i, j, h_quad_pts(i, j));
    }

    }
}

inline
void InteriorFaceHelperFunctor::get_quadrature(
    Basis::Basis basis, const int order){

    // unpack
    int NDIMS = basis.face_shape.get_NDIMS();
    int nq_1d; int nq;
    int qorder = basis.face_shape.get_quadrature_order(order);

    QuadratureTools::get_number_of_quadrature_points(qorder, NDIMS,
            nq_1d, nq);

    printf("face qorder=%i", qorder);
    // need to establish an initial size for views prior
    // to resizing inside of get_quadrature_data
    Kokkos::resize(quad_pts, nq, NDIMS);
    Kokkos::resize(quad_wts, nq);

    h_quad_pts = Kokkos::create_mirror_view(quad_pts);
    h_quad_wts = Kokkos::create_mirror_view(quad_wts);

    basis.face_shape.get_quadrature_data(qorder, nq_1d, h_quad_pts, h_quad_wts);

    Kokkos::deep_copy(quad_pts, h_quad_pts);
    Kokkos::deep_copy(quad_wts, h_quad_wts);

}

} // end namespace InteriorFaceHelpers

