#include <vector>

namespace Basis {

/* --------------------------------------
        Segment Shape Definitions
----------------------------------------*/
inline
int get_num_basis_coeff_segment(int p){
    return p + 1;
}

inline
unsigned get_num_nodes_per_face_segment(const unsigned gorder) {
    return 1;
}

inline
unsigned get_num_nodes_per_elem_segment(const unsigned gorder) {
    return (gorder+1);
}

/* --------------------------------------
        Quadrilateral Shape Definitions
----------------------------------------*/
inline
int get_num_basis_coeff_quadrilateral(int p){
    return (p + 1) * (p + 1);
}

inline
unsigned get_num_nodes_per_face_quadrilateral(const unsigned gorder){
    return gorder + 1;
}

inline
unsigned get_num_nodes_per_elem_quadrilateral(const unsigned gorder){
    return (gorder + 1) * (gorder + 1);
}

inline //KOKKOS_INLINE_FUNCTION
void get_face_pts_order_wrt_orient0_quadrilateral(const int orient, const int npts,
        Kokkos::View<int*, Kokkos::OpenMP> pts_order) {

    assert(npts == (int) pts_order.extent(0));
    switch (orient) {
        case 0: {
            for (int i=0; i<npts; i++) {
                pts_order(i) = i;
            }
            break;
        }

        case 1: {
            for (int i=0; i<npts; i++) {
                pts_order(i) = npts - 1 - i;
            }
            break;
        }

        default: {
            printf("FATAL EXCEPTION IN 'get_face_pts_order_wrt_orient0_quadrilateral'");
        }
    }
}

inline
void get_points_on_face_quadrilateral(const int face_id, const int orient, const int np,
        const Kokkos::View<rtype**>::HostMirror face_pts,
        Kokkos::View<rtype**, Kokkos::LayoutStride>::HostMirror elem_pts){

    // Positions of all nodes
    const std::vector<rtype> xref_nodes{-1, 1., -1, 1.};
    const std::vector<rtype> yref_nodes{-1, -1., 1., 1.};
    std::vector<int> node(2, 0);

    // Determine which nodes are on the desired face
    switch (face_id) {
        case 0:
            node[0] = 0;
            node[1] = 1;
            break;
        case 1:
            node[0] = 1;
            node[1] = 3;
            break;
        case 2:
            node[0] = 3;
            node[1] = 2;
            break;
        case 3:
            node[0] = 2;
            node[1] = 0;
            break;
        default:
            std::string error_message = "Face ID ";
            error_message += std::to_string(face_id);
            error_message += " not available for quadrilateral";
            throw std::runtime_error(error_message);
    }

    if (orient == 1) {
        std::swap(node[0], node[1]);
    }

    rtype x0, x1, y0, y1, b0, b1;
    x0 = xref_nodes[node[0]];
    y0 = yref_nodes[node[0]];
    x1 = xref_nodes[node[1]];
    y1 = yref_nodes[node[1]];

    for (int i = 0; i < np; i++) {
        // barycentric coordinates
        b1 = (face_pts(i, 0) + 1.) / 2.;
        b0 = 1.0 - b1;

        elem_pts(i, 0) = (b0 * x0 + b1 * x1);
        elem_pts(i, 1) = (b0 * y0 + b1 * y1);
    }
}


KOKKOS_INLINE_FUNCTION
void get_local_nodes_on_face_quadrilateral(const int face_id, const int gorder,
    scratch_view_1D_int lfnodes) {

        int i0 = 0, d = 0;
        
        switch (face_id) {
        case 0:
            i0 = 0;
            d = 1;
            break;
        case 1:
            i0 = gorder;
            d = gorder + 1;
            break;
        case 2:
            i0 = gorder * (gorder + 2);
            d = -1;
            break;
        case 3:
            i0 = gorder * (gorder + 1);
            d = -gorder - 1;
            break;
    }

    for (int i = 0; i < gorder + 1; i++) {
        lfnodes(i) = i0 + i * d;
    }
}

template<typename ViewType2D, typename ViewType3D_gbasis_grad, 
typename ViewType3D_xphys_grad,
typename ViewType3D_normals, typename MemberType> KOKKOS_INLINE_FUNCTION
void get_normals_on_face_quadrilateral(const int np, const int gorder,
        const ViewType3D_gbasis_grad face_gbasis_ref_grad, const ViewType2D face_coords,
        ViewType3D_xphys_grad xphys_grad, ViewType3D_normals normals, const MemberType& member) {

        const int iface = member.league_rank();

        for (unsigned ip = 0; ip < np; ip++){
            auto face_gbasis_ref_grad_ = 
                Kokkos::subview(face_gbasis_ref_grad, ip, Kokkos::ALL, Kokkos::ALL);
            auto xphys_grad_ = Kokkos::subview(xphys_grad, Kokkos::ALL, Kokkos::ALL, ip);
            // Math::cAxB_to_C(1., face_gbasis_ref_grad_, face_coords, xphys_grad, member);

            Math::cATxB_to_C(1., face_coords, face_gbasis_ref_grad_, xphys_grad_, member);

            // if (iface == 3){
            //     for (unsigned i = 0; i < face_coords.extent(0); i++){
            //         for (unsigned j = 0; j < face_coords.extent(1); j++){
            //             printf("iface=%i -> face_coords(%i, %i)=%f\n", iface, i, j, face_coords(i, j));
            //         }
            //     }
            //     for (unsigned i = 0; i < face_gbasis_ref_grad_.extent(0); i++){
            //         for (unsigned j = 0; j < face_gbasis_ref_grad_.extent(1); j++){
            //             printf("iface=%i -> face_gbasis_ref_grad_(%i, %i)=%f\n", iface, i, j, face_gbasis_ref_grad_(i, j));
            //         }
            //     }
            //     for (int i = 0; i < xphys_grad.extent(0); i++){
            //         for (int j = 0; j < xphys_grad.extent(1); j++){
            //             printf("iface=%i -> xphys_grad(%i, %i)=%f\n", member.league_rank(), i, j, xphys_grad(i, j));
            //         }
            //     }
            // }
        }
        // for (int i = 0; i < xphys_grad.extent(0); i++){
        //     for (int j = 0; j < xphys_grad.extent(1); j++){
        //         printf("iface=%i -> xphys_grad(%i, %i)=%f\n", member.league_rank(), i, j, xphys_grad(i, j));
        //     }
        // }

        for (unsigned ip = 0; ip < np; ip++){
            normals(iface, ip, 0) = xphys_grad(1, 0, ip);
            normals(iface, ip, 1) = -1.0 * xphys_grad(0, 0, ip);
        }
        

}

/* --------------------------------------
        Hexahedron Shape Definitions
----------------------------------------*/
inline
int get_num_basis_coeff_hexahedron(int p){
    return (p + 1) * (p + 1) * (p + 1);
}

inline
unsigned get_num_nodes_per_face_hexahedron(const unsigned gorder){
    return (gorder + 1) * (gorder + 1);
}

inline
unsigned get_num_nodes_per_elem_hexahedron(const unsigned gorder){
    return (gorder + 1) * (gorder + 1) * (gorder + 1);
}

inline
void get_points_on_face_hexahedron(const int face_id, const int orient, const int np,
        const Kokkos::View<rtype**>::HostMirror face_pts,
        Kokkos::View<rtype**, Kokkos::LayoutStride>::HostMirror elem_pts){

    // All nodes in element
    const std::vector<rtype> xref_nodes{-1., 1.,-1., 1.,-1., 1.,-1., 1.};
    const std::vector<rtype> yref_nodes{-1.,-1., 1., 1.,-1.,-1., 1., 1.};
    const std::vector<rtype> zref_nodes{-1.,-1.,-1.,-1., 1., 1., 1., 1.};
    std::vector<int> node_temp(4, 0);
    std::vector<int> node(4, 0);

    switch (face_id) {
        case 0:
            node_temp[0] = 0;
            node_temp[1] = 2;
            node_temp[2] = 3;
            node_temp[3] = 1;
            break;
        case 1:
            node_temp[0] = 0;
            node_temp[1] = 1;
            node_temp[2] = 5;
            node_temp[3] = 4;
            break;
        case 2:
            node_temp[0] = 1;
            node_temp[1] = 3;
            node_temp[2] = 7;
            node_temp[3] = 5;
            break;
        case 3:
            node_temp[0] = 3;
            node_temp[1] = 2;
            node_temp[2] = 6;
            node_temp[3] = 7;
            break;
        case 4:
            node_temp[0] = 2;
            node_temp[1] = 0;
            node_temp[2] = 4;
            node_temp[3] = 6;
            break;
        case 5:
            node_temp[0] = 4;
            node_temp[1] = 5;
            node_temp[2] = 7;
            node_temp[3] = 6;
            break;
        default:
            std::string error_message = "Face ID ";
            error_message += std::to_string(face_id);
            error_message += " not available for hexahedron";
            throw std::runtime_error(error_message);
    }

    int cycle, flip;
    cycle = orient%4;
    flip = orient/4;

    // Orient nodes correctly
    for (int j = 0; j < 4; j++) {
        node[j] = node_temp[(j+cycle)%4];
    }
    if (flip >= 1) {
        std::swap(node[1],node[3]);
    }

    std::vector<rtype> xelem(4,0.0);
    std::vector<rtype> yelem(4,0.0);
    std::vector<rtype> zelem(4,0.0);
    std::vector<rtype> b(4,0.0);
    rtype xface_bar, yface_bar;

    for (int j = 0; j < 4; j++) {
        xelem[j] = xref_nodes[node[j]];
        yelem[j] = yref_nodes[node[j]];
        zelem[j] = zref_nodes[node[j]];
    }

    for (int i = 0; i < np; i++) {
        // barycentric coordinates
        xface_bar = (face_pts(i, 0) + 1.)/2.;
        yface_bar = (face_pts(i, 1) + 1.)/2.;

        b[0] = (1.-xface_bar)*(1.-yface_bar);
        b[1] = xface_bar*(1.-yface_bar);
        b[2] = xface_bar*yface_bar;
        b[3] = (1.-xface_bar)*yface_bar;

        elem_pts(i, 0) = 0.0;
        elem_pts(i, 1) = 0.0;
        elem_pts(i, 2) = 0.0;

        for (int j = 0; j < 4; j++){
            elem_pts(i, 0) += b[j]*xelem[j];
            elem_pts(i, 1) += b[j]*yelem[j];
            elem_pts(i, 2) += b[j]*zelem[j];
        }
    }
}

inline //KOKKOS_INLINE_FUNCTION
void get_face_pts_order_wrt_orient0_hexahedron(const int orient, const int npts,
        Kokkos::View<int*, Kokkos::OpenMP> pts_order) {

    assert(npts == (int) pts_order.extent(0));
    assert(npts > 0);
    const int n1D = Kokkos::Experimental::sqrt(npts);
    assert(n1D*n1D == npts); // a hexahedron's face is a quadrilateral
    const int nptsp1 = npts + 1;

    switch (orient) {
        case 0: {
            for (unsigned i=0; i<(unsigned)npts; i++) {
                pts_order(i) = i;
            }
            break;
        }

        case 1: {
            int curr = n1D - 1;
            for (unsigned i=0; i<(unsigned)npts; i++) {
                pts_order(i) = curr;
                curr = (curr + n1D) % nptsp1;
            }
            break;
        }

        case 2: {
            for (unsigned i=0; i<(unsigned)npts; i++) {
                pts_order(i) = npts-1-i;
            }
            break;
        }

        case 3: {
            int curr = npts - n1D;
            for (unsigned i=0; i<(unsigned)npts; i++) {
                pts_order(i) = curr;
                curr -= n1D;
                if (curr < 0) {
                    curr += nptsp1;
                }
            }
            break;
        }

        case 4: {
            int curr = 0;
            for (unsigned i=0; i<(unsigned)npts; i++) {
                pts_order(i) = curr;
                curr += n1D;
                if (curr >= npts) {
                    curr -= npts - 1;
                }
            }
            break;
        }

        case 5: {
            int curr = n1D - 1;
            unsigned k = 0;
            for (unsigned i=0; i<(unsigned)n1D; i++) {
                for (unsigned j=0; j<(unsigned)n1D; j++) {
                    pts_order(k) = curr;
                    curr -= 1;
                    k += 1;
                }
                curr += 2*n1D;
            }
            break;
        }

        case 6: {
            int curr = npts - 1;
            for (unsigned i=0; i<(unsigned)npts; i++) {
                pts_order(i) = curr;
                curr -= n1D;
                if (curr < 0) {
                    curr += npts-1;
                }
            }
            break;
        }

        case 7: {
            int curr = npts - n1D;
            unsigned k = 0;
            for (unsigned i=0; i<(unsigned)n1D; i++) {
                for (unsigned j=0; j<(unsigned)n1D; j++) {
                    pts_order(k) = curr;
                    curr += 1;
                    k += 1;
                }
                curr -= 2*n1D;
            }
            break;
        }

        default: {
            printf("FATAL EXCEPTION");
        }
    }
}

KOKKOS_INLINE_FUNCTION
void get_local_nodes_on_face_hexahedron(const int face_id, const int gorder,
    scratch_view_1D_int lfnodes) {

    int i0=0, d0=0, d1=0;

    switch (face_id) {
        case 0:
            // bottom
            i0 = 0;
            d0 = gorder+1;
            d1 = 1;
            break;
        case 1:
            // front
            i0 = 0;
            d0 = 1;
            d1 = (gorder+1)*(gorder+1);
            break;
        case 2:
            // right
            i0 = gorder;
            d0 = gorder+1;
            d1 = (gorder+1)*(gorder+1);
            break;
        case 3:
            // back
            i0 = (gorder+1)*(gorder+1)-1;
            d0 = -1;
            d1 = (gorder+1)*(gorder+1);
            break;
        case 4:
            // left
            i0 = gorder*(gorder+1);
            d0 = -(gorder+1);
            d1 = (gorder+1)*(gorder+1);
            break;
        case 5:
            // top
            i0 = gorder*(gorder+1)*(gorder+1);
            d0 = 1;
            d1 = gorder+1;
            break;
    }

    for (int j = 0, k = 0; j < gorder+1; j++){
        for (int i = 0; i < gorder+1; i++, k++) {
            lfnodes(k) = i0 + i*d0 + j*d1;
        }
    }
}

template<typename ViewType2D, typename ViewType3D_gbasis_grad,
typename ViewType3D_xphys_grad,
typename ViewType3D_normals, typename MemberType> KOKKOS_INLINE_FUNCTION
void get_normals_on_face_hexahedron(const int np, const int gorder,
        const ViewType3D_gbasis_grad face_gbasis_ref_grad, const ViewType2D face_coords,
        ViewType3D_xphys_grad xphys_grad, ViewType3D_normals normals, const MemberType& member) {


        const int iface = member.league_rank();

        for (unsigned ip = 0; ip < np; ip++){
            auto face_gbasis_ref_grad_ = 
                Kokkos::subview(face_gbasis_ref_grad, ip, Kokkos::ALL, Kokkos::ALL);
            auto xphys_grad_ = Kokkos::subview(xphys_grad, Kokkos::ALL, Kokkos::ALL, ip);

            // Math::cAxB_to_C(1., face_gbasis_ref_grad_, face_coords, xphys_grad, member);
            Math::cATxB_to_C(1., face_coords, face_gbasis_ref_grad_, xphys_grad_, member);

        }

        // if (iface == 3){
        //         for (int i = 0; i < xphys_grad.extent(0); i++){
        //             for (int j = 0; j < xphys_grad.extent(1); j++){
        //                 for (int k = 0; k < xphys_grad.extent(2); k++){
        //                 printf("iface=%i -> xphys_grad(%i, %i, %i)=%f\n", member.league_rank(), i, j, k, xphys_grad(i, j, k));
        //             }
        //             }
        //         }
        // }
        // for (int i = 0; i < xphys_grad.extent(0); i++){
        //     for (int j = 0; j < xphys_grad.extent(1); j++){
        //         printf("iface=%i -> xphys_grad(%i, %i)=%f\n", member.league_rank(), i, j, xphys_grad(i, j));
        //     }
        // }
        // for (int i = 0; i < face_coords.extent(0); i++){
        //     for (int j = 0; j < face_coords.extent(1); j++){
        //         printf("iface=%i -> face_coords(%i, %i)=%f\n", member.league_rank(), i, j, face_coords(i, j));
        //     }
        // }

        rtype x_xref1[3], x_xref2[3];
        for (unsigned ip = 0; ip < np; ip++){
            for (unsigned j = 0; j < 3; j++){
                x_xref1[j] = xphys_grad(j, 0, ip);
                x_xref2[j] = xphys_grad(j, 1, ip);
            }
            auto normals_ = Kokkos::subview(normals, iface, ip, Kokkos::ALL);
            Math::cross(x_xref1, x_xref2, normals_);
        }
}

inline
Shape::Shape(ShapeType shape_type){
    if (shape_type == ShapeType::Segment){
        // set methods
        get_num_basis_coeff = get_num_basis_coeff_segment;
        get_quadrature_order_pointer =
            SegmentQuadrature::get_gausslegendre_quadrature_order;
        get_quadrature_data =
            SegmentQuadrature::get_quadrature_gauss_legendre;
        get_num_nodes_per_face = get_num_nodes_per_face_segment;
        get_num_nodes_per_elem = get_num_nodes_per_elem_segment;
        // set constants
        NDIMS = 1;
        NFACES = 2;
        NCORNERS = 2;
        NUM_ORIENT_PER_FACE = 1;
        name = "Segment";
        type = ShapeType::Segment;
    }
    if (shape_type == ShapeType::Quadrilateral){
        // set methods
        get_num_basis_coeff = get_num_basis_coeff_quadrilateral;
        get_quadrature_order_pointer =
            QuadrilateralQuadrature::get_gausslegendre_quadrature_order;
        get_quadrature_data =
            QuadrilateralQuadrature::get_quadrature_gauss_legendre;
        get_points_on_face = get_points_on_face_quadrilateral;
        get_num_nodes_per_face = get_num_nodes_per_face_quadrilateral;
        get_num_nodes_per_elem = get_num_nodes_per_elem_quadrilateral;

        // set constants
        NDIMS = 2;
        NFACES = 4;
        NCORNERS = 4;
        NUM_ORIENT_PER_FACE = 2;
        name = "Quadrilateral";
        type = ShapeType::Quadrilateral;
        face_type = ShapeType::Segment;
    }
    if (shape_type == ShapeType::Hexahedron){
        // set methods
        get_num_basis_coeff = get_num_basis_coeff_hexahedron;
        get_quadrature_order_pointer =
            QuadratureTools::get_gausslegendre_quadrature_order;
        get_quadrature_data =
            HexahedronQuadrature::get_quadrature_gauss_legendre;
        get_points_on_face = get_points_on_face_hexahedron;
        get_num_nodes_per_face = get_num_nodes_per_face_hexahedron;
        get_num_nodes_per_elem = get_num_nodes_per_elem_hexahedron;

        // set constants
        NDIMS = 3;
        NCORNERS = 8;
        NFACES = 6;
        NUM_ORIENT_PER_FACE = 8;
        name = "Hexahedron";
        type = ShapeType::Hexahedron;
        face_type = ShapeType::Quadrilateral;

    }
}

inline //KOKKOS_INLINE_FUNCTION
void Shape::get_face_pts_order_wrt_orient0(const int orient, const int npts,
        Kokkos::View<int*, Kokkos::OpenMP> pts_order) const {
    if (type == ShapeType::Quadrilateral){
        get_face_pts_order_wrt_orient0_quadrilateral(orient, npts, pts_order);
    }
    else if (type == ShapeType::Hexahedron){
        get_face_pts_order_wrt_orient0_hexahedron(orient, npts, pts_order);    
    } else {
        printf("Not Implemented Error\n"); // TODO: Figure out GPU throws
    }
}

KOKKOS_INLINE_FUNCTION
void Shape::get_local_nodes_on_face(const int face_id, const int gorder,
    scratch_view_1D_int lfnodes) const {

    if (type == ShapeType::Quadrilateral) {
        get_local_nodes_on_face_quadrilateral(face_id, gorder, lfnodes);
    }
    else if (type == ShapeType::Hexahedron){
        get_local_nodes_on_face_hexahedron(face_id, gorder, lfnodes);
    } else {
        printf("Not Implemented Error\n"); // TODO: Figure out GPU throws
    }
}

template<typename ViewType2D, typename ViewType3D_gbasis_grad, 
typename ViewType3D_xphys_grad, typename ViewType3D_normals, 
typename MemberType> KOKKOS_INLINE_FUNCTION
void Shape::get_normals_on_face(const int np, const int gorder,
        const ViewType3D_gbasis_grad face_gbasis_ref_grad,
        const ViewType2D face_coords, ViewType3D_xphys_grad xphys_grad,
        ViewType3D_normals normals, const MemberType& member) const {

    if (type == ShapeType::Quadrilateral) {
        get_normals_on_face_quadrilateral(np, gorder, face_gbasis_ref_grad,
            face_coords, xphys_grad, normals, member);
    }
    else if (type == ShapeType::Hexahedron){
        get_normals_on_face_hexahedron(np, gorder, face_gbasis_ref_grad,
            face_coords, xphys_grad, normals, member);    
    } 
    else {
        printf("Normals Not Implemented Error\n"); // TODO: Figure out GPU throws
    }


}

inline
int Shape::get_quadrature_order(const int order_) {
    return get_quadrature_order_pointer(order_, NDIMS);
}

} // end namespace Basis
