namespace VolumeHelpers {


inline
void VolumeHelperFunctor::compute_volume_helpers(int scratch_size,
    Mesh& mesh, Basis::Basis& basis, MemoryNetwork& network){

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
    get_reference_data(basis, mesh.gbasis, basis.get_order());
    precompute_facequadrature_lookup(mesh, basis);
    precompute_normals(mesh, basis);
    // TODO: Add face ijac computation
}

inline
void InteriorFaceHelperFunctor::get_quadrature(
    Basis::Basis basis, const int order){

    // unpack
    const unsigned NFACE = basis.shape.get_num_faces_per_elem();
    const unsigned NDIMS_FACE = basis.face_shape.get_NDIMS();
    const unsigned NDIMS = basis.shape.get_NDIMS();

    int nq_1d; int nq;
    int qorder = basis.face_shape.get_quadrature_order(order);
    QuadratureTools::get_number_of_quadrature_points(qorder, NDIMS_FACE,
            nq_1d, nq);

    // quad_wts is the same for all faces (i.e. orientation and specific face doesn't matter)
    Kokkos::resize(quad_wts, nq);
    h_quad_wts = Kokkos::create_mirror_view(quad_wts);

    // quad_pts needs to know the specific face
    view_type_2D local_quad_pts("local quad_pts", nq, NDIMS_FACE);
    host_view_type_2D h_local_quad_pts = Kokkos::create_mirror_view(local_quad_pts);
    // get the quad_pts / quad_wts for a reference face
    basis.face_shape.get_quadrature_data(qorder, nq_1d, h_local_quad_pts, h_quad_wts);

    Kokkos::resize(quad_pts, NFACE, nq, NDIMS);
    h_quad_pts = Kokkos::create_mirror_view(quad_pts);

    // loop over shape faces per face and get the points on that face
    // for a given shape object
    for (unsigned ifa = 0; ifa < NFACE; ifa++){
        // we pass in zero for the orientation as we only store the zero orientation
        basis.shape.get_points_on_face(ifa, 0, nq, h_local_quad_pts,
            Kokkos::subview(h_quad_pts, ifa, Kokkos::ALL(), Kokkos::ALL()));
    }

    // for (unsigned ifa = 0; ifa < NFACE; ifa++){
    //         for (unsigned iq = 0; iq < h_quad_pts.extent(1); iq++){
    //             for (unsigned i = 0; i < h_quad_pts.extent(2); i++){
    //             printf("h_quad_pts(%i, %i, %i)=%f\n", ifa, iq, i, h_quad_pts(ifa, iq, i));
    //         }
    //     }
    // }


    Kokkos::deep_copy(quad_pts, h_quad_pts);
    Kokkos::deep_copy(quad_wts, h_quad_wts);

}

inline
void InteriorFaceHelperFunctor::get_reference_data(
        Basis::Basis basis, Basis::Basis gbasis, const int order){

    // unpack
    const unsigned NFACE = basis.shape.get_num_faces_per_elem();
    const unsigned NDIMS_FACE = basis.face_shape.get_NDIMS();
    const unsigned NDIMS = basis.shape.get_NDIMS();

    int nb = basis.shape.get_num_basis_coeff(order);
    int gnb = gbasis.shape.get_num_basis_coeff(gbasis.get_order());
    int nq = quad_pts.extent(1);

    // need to establish an initial size for views
    Kokkos::resize(basis_val, NFACE, nq, nb);
    Kokkos::resize(basis_ref_grad, NFACE, nq, nb, NDIMS);
    Kokkos::resize(gbasis_val, NFACE, nq, gnb);
    Kokkos::resize(gbasis_ref_grad, NFACE, nq, gnb, NDIMS);

    h_basis_val = Kokkos::create_mirror_view(basis_val);
    h_basis_ref_grad = Kokkos::create_mirror_view(basis_ref_grad);
    h_gbasis_val = Kokkos::create_mirror_view(gbasis_val);
    h_gbasis_ref_grad = Kokkos::create_mirror_view(gbasis_ref_grad);

    for (unsigned ifa = 0; ifa < NFACE; ifa++){

        // get subviews prior to passing into get_values and get_grads
        auto h_quad_pts_ = Kokkos::subview(h_quad_pts, ifa, Kokkos::ALL(), Kokkos::ALL());
        auto h_basis_val_ = Kokkos::subview(h_basis_val, ifa, Kokkos::ALL(), Kokkos::ALL());
        auto h_gbasis_val_ = Kokkos::subview(h_gbasis_val, ifa, Kokkos::ALL(), Kokkos::ALL());
        auto h_basis_ref_grad_ = Kokkos::subview(h_basis_ref_grad, ifa, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());
        auto h_gbasis_ref_grad_ = Kokkos::subview(h_gbasis_ref_grad, ifa, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());

        basis.get_values(h_quad_pts_, h_basis_val_);
        gbasis.get_values(h_quad_pts_, h_gbasis_val_);

        basis.get_grads(h_quad_pts_, h_basis_ref_grad_);
        gbasis.get_grads(h_quad_pts_, h_gbasis_ref_grad_);

        // for (unsigned i = 0; i < h_basis_val_.extent(0); i++){
        // for (unsigned j = 0; j < h_basis_val_.extent(1); j++){
        //     printf("ifa: %i, phi(%i, %i)=%f\n", ifa, i, j, h_basis_val(ifa, i, j));
        // }
        // }

        // for (unsigned i = 0; i < h_gbasis_val_.extent(0); i++){
        // for (unsigned j = 0; j < h_gbasis_val_.extent(1); j++){
        //     printf("ifa: %i, ior: %i, gphi(%i, %i)=%f\n", ifa, ior, i, j, h_gbasis_val(ifa, ior, i, j));
        // }
        // }

    }

    Kokkos::deep_copy(basis_val, h_basis_val);
    Kokkos::deep_copy(basis_ref_grad, h_basis_ref_grad);

    Kokkos::deep_copy(gbasis_val, h_gbasis_val);
    Kokkos::deep_copy(gbasis_ref_grad, h_gbasis_ref_grad);

}


inline
void InteriorFaceHelperFunctor::precompute_facequadrature_lookup(Mesh& mesh,
    Basis::Basis basis){

    const int nqf = quad_pts.extent(1);

    Kokkos::View<int*, Kokkos::OpenMP> orderL("orderL", nqf);
    Kokkos::View<int*, Kokkos::OpenMP> orderR("orderR", nqf);

    Kokkos::resize(quad_idx_L, mesh.num_ifaces_part, nqf);
    Kokkos::resize(quad_idx_R, mesh.num_ifaces_part, nqf);

    // TODO: Remove when made parallel
    Kokkos::View<int**>::HostMirror h_quad_idx_L = Kokkos::create_mirror_view_and_copy(
            Kokkos::DefaultHostExecutionSpace{}, quad_idx_L);
    Kokkos::View<int**>::HostMirror h_quad_idx_R = Kokkos::create_mirror_view_and_copy(
            Kokkos::DefaultHostExecutionSpace{}, quad_idx_R);

    // Kokkos::parallel_for(mesh.num_ifaces_part, KOKKOS_CLASS_LAMBDA(const int& iface){
    // TODO: Figure out why there is a race condition here when using parallel_for...
    for (int iface = 0; iface < mesh.num_ifaces_part; iface++){
        const unsigned orientL = mesh.get_orientL_host(iface);
        const unsigned orientR = mesh.get_orientR_host(iface);

        // if (iface == 0){
        // printf("oL=%i\n", orientL);
        // printf("oR=%i\n", orientR);
        // }
        // printf("oL=%i\n", orientL);
        // printf("oR=%i\n", orientR);
        // Kokkos::fence();

        basis.shape.get_face_pts_order_wrt_orient0(orientL, nqf, orderL);
        basis.shape.get_face_pts_order_wrt_orient0(orientR, nqf, orderR);

        // Kokkos::fence();
        for (unsigned iq = 0; iq < nqf; iq++){
            h_quad_idx_L(iface, iq) = orderL(iq);
            h_quad_idx_R(iface, iq) = orderR(iq);

            // printf("quad_idx_L(%i, %i)=%i\n", iface, iq, orderL(iq));
            // printf("quad_idx_R(%i, %i)=%i\n", iface, iq, orderR(iq));

        }
        // Kokkos::fence();

    }
    Kokkos::deep_copy(quad_idx_L, h_quad_idx_L);
    Kokkos::deep_copy(quad_idx_R, h_quad_idx_R);

    // printf("quad_idx_L(0, 0)=%i\n", quad_idx_L(0, 0));
    // printf("quad_idx_L(0, 1)=%i\n", quad_idx_L(0, 1));
}

inline
void InteriorFaceHelperFunctor::precompute_normals(Mesh& mesh, Basis::Basis basis){

    // Create memory network
    auto network = MemoryNetwork();

    const int nqf = quad_pts.extent(1);
    const unsigned gorder = mesh.gbasis.get_order();
    const unsigned num_nodes_per_face = mesh.gbasis.shape.get_num_nodes_per_face(gorder);
    const unsigned num_nodes_per_elem = mesh.gbasis.shape.get_num_nodes_per_elem(gorder);

    // resize the normal view accordingly
    Kokkos::resize(normals, mesh.num_ifaces_part, nqf, mesh.dim);

    // allocate bytes for calculating the precompute normals kernel
    int scratch_size_normals =
        scratch_view_1D_int::shmem_size(num_nodes_per_face) +
        scratch_view_2D_rtype::shmem_size(num_nodes_per_face, mesh.dim) +
        scratch_view_2D_rtype::shmem_size(num_nodes_per_elem, mesh.dim) +
        scratch_view_3D_rtype::shmem_size(mesh.dim, (mesh.dim - 1), nqf);

    // we need the reference space gradient of the geometric face basis later
    // in our face normal calculation but we choose to do this here so we can
    // capture the face_basis_ref_grad into the Kokkos lambda and not have to
    // pass it around using scratch memory
    view_type_3D face_gbasis_ref_grad =
        mesh.gbasis.get_face_basis_ref_grad_for_normals(gorder, h_quad_pts);

    Kokkos::parallel_for("normals", Kokkos::TeamPolicy<>( mesh.num_ifaces_part,
        Kokkos::AUTO).set_scratch_size( 1, Kokkos::PerTeam( scratch_size_normals )),
        KOKKOS_CLASS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& member) {
        // get current iface
        const unsigned iface = member.league_rank();
        // face normals are chosen to point from the left element to the right,
        // or outward wrt the left element (i.e. -> we always use the left facing data)
        const unsigned face_ID_L = mesh.get_ref_face_idL(iface);
        const unsigned face_ID_R = mesh.get_ref_face_idR(iface);
        // Get global element IDs on either side
        const unsigned global_elemL = mesh.get_elemL(iface);
        const unsigned global_elemR = mesh.get_elemR(iface);
        // Convert to local
        auto elemL = mesh.search_for_local_ID(global_elemL, mesh.local_to_global_elem_IDs);
        auto elemR = mesh.search_for_local_ID(global_elemR, mesh.local_to_global_elem_IDs);
        // get allocation from scratch memory
        scratch_view_1D_int face_node_idx(member.team_scratch( 1 ),
            num_nodes_per_face);
        scratch_view_2D_rtype face_coord(member.team_scratch( 1 ),
            num_nodes_per_face, mesh.dim);
        scratch_view_2D_rtype elem_coords(member.team_scratch( 1 ),
            num_nodes_per_elem, mesh.dim);
        scratch_view_3D_rtype xphys_grad(member.team_scratch( 1 ),
            mesh.dim, (mesh.dim - 1), nqf);

        rtype sign;
        // TODO: Currently this doesn't work for multi-rank cases
        // To fix this we can set up a view that contains the face coordinates
        // for each interior face. This would remove the need for elem_coords
        // and directly store face_coords for each iface
        // if (member.team_rank() == 0 ) {
        unsigned elem_ID;
        unsigned ref_face_ID;
        // If this rank has info on the left of the face, take the nodes
        // from the left element
        if (mesh.interior_faces(iface, 0) == network.rank) {
            elem_ID = elemL;
            ref_face_ID = face_ID_L;
            sign = 1.0;
        // If this rank has info on the right of the face, take the nodes
        // from the right element
        } else if (mesh.interior_faces(iface, 4) == network.rank) {
            elem_ID = elemR;
            ref_face_ID = face_ID_R;
            sign = -1.0;
        // This branch should, hopefully, never happen
        } else {
            printf("Nooooo! Bad things happened!\n");
            printf("(this face not found on this rank when precomputing normals)\n");
        }
        // printf("%u %u\n", elem_ID, ref_face_ID);
        MeshTools::elem_coords_from_elem_ID(mesh, elem_ID, elem_coords, member);
        // populate face_node_idx which is filled with reference node id wrt
        // the reference face ID number (Ex: 0-3 for quadrilaterals) and left element ID
        mesh.gbasis.shape.get_local_nodes_on_face(ref_face_ID, gorder, face_node_idx);
        // extract the face coordinates from the mesh coordinates
        BasisTools::extract_node_coordinates(mesh.dim, elem_coords, face_node_idx, face_coord);
    // }
        member.team_barrier();



        // for (unsigned i = 0; i < face_coord.extent(0); i++){
        //     for (unsigned j = 0; j < face_coord.extent(1); j++){
        //         printf("iface=%i -> face_coord(%i, %i)=%f\n", iface, i, j, face_coord(i, j));
        //     }
        // }
        member.team_barrier();
        // for (unsigned i = 0 ; i < num_nodes_per_face; i++){
        //     printf("face_node_idx(%i)=%i\n", i, face_node_idx(i));
        // }

        // get the normals for each iface
        mesh.gbasis.shape.get_normals_on_face(sign, (int)nqf, (int)gorder,
            face_gbasis_ref_grad, face_coord, xphys_grad, normals, member);

        // int global_face_ID = mesh.get_global_iface_ID(iface);
        // if (global_face_ID == 5){
            // printf("face left rank = %i\n", mesh.interior_faces(iface, 0));
            // printf("face right rank = %i\n", mesh.interior_faces(iface, 4));
            // printf("sign=%f\n", sign);
        // for (unsigned i = 0; i < normals.extent(1); i++){
        //     for (unsigned j = 0; j < normals.extent(2); j++){
        //         if ( mesh.interior_faces(iface, 0) != mesh.interior_faces(iface, 4)){
        //         printf("Rank=%i, LRank=%i, RRank=%i, sign=%f, normals(%i, %i, %i)=%f\n", 
        //             network.rank, mesh.interior_faces(iface, 0), mesh.interior_faces(iface, 4), 
        //             sign, global_face_ID, i, j, normals(iface, i, j));
        //         }
        //     }
        // }
        // }

    });
}

} // end namespace InteriorFaceHelpers

