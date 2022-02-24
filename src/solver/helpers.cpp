namespace VolumeHelpers {

VolumeHelperFunctor::VolumeHelperFunctor(Mesh mesh, Basis::Basis basis)
    : mesh{mesh}, basis{basis} {}


void VolumeHelperFunctor::compute_volume_helpers(int scratch_size){

    get_quadrature(basis, basis.get_order());
    get_reference_data(basis, mesh.gbasis, basis.get_order());
    allocate_views(mesh, mesh.num_elems_part);

    Kokkos::parallel_for("volume helpers", Kokkos::TeamPolicy<VolumeHelperTag>( mesh.num_elems_part,
            Kokkos::AUTO ).set_scratch_size( 0,
            Kokkos::PerThread( scratch_size )), *this);
}

void VolumeHelperFunctor::compute_inv_mass_matrices(int scratch_size){

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

    Kokkos::parallel_for("iMM helper", Kokkos::TeamPolicy<iMMHelperTag>( mesh.num_elems_part,
            Kokkos::AUTO ).set_scratch_size( 1,
            Kokkos::PerTeam( scratch_size )), *this);
}


KOKKOS_INLINE_FUNCTION
void VolumeHelperFunctor::operator()(VolumeHelperTag, const member_type& member) const {
    // Fetch the index of the calling team within the league
    const int elem_ID = member.league_rank();

    scratch_view_2D_rtype elem_coords(member.team_scratch( 0 ),
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

}


KOKKOS_INLINE_FUNCTION
void VolumeHelperFunctor::operator()(iMMHelperTag, const member_type& member) const {
    // Fetch the index of the calling team within the league
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


}

void VolumeHelperFunctor::allocate_views(Mesh& mesh, const int num_elems){

    const int nq = quad_pts.extent(0);
    const int ndims = quad_pts.extent(1);

    Kokkos::resize(jac_elems, num_elems, nq, ndims, ndims);
    Kokkos::resize(djac_elems, num_elems, nq);
    Kokkos::resize(ijac_elems, num_elems, nq, ndims, ndims);
    Kokkos::resize(x_elems, num_elems, nq, ndims);
    Kokkos::resize(vol_elems, num_elems);
}

void VolumeHelperFunctor::get_quadrature(
    Basis::Basis basis, const int order){

    // unpack
    int NDIMS = basis.shape.get_NDIMS();
    int qorder = basis.shape.get_quadrature_order(order);
    int nq_1d; int nq;
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



} // end namespace VolumeHelper
