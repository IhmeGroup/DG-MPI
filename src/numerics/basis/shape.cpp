#include <vector>

namespace Basis {

/* --------------------------------------
        Segment Shape Definitions
----------------------------------------*/
inline
int get_num_basis_coeff_segment(int p){
    return p + 1;
}


/* --------------------------------------
        Quadrilateral Shape Definitions
----------------------------------------*/
inline
int get_num_basis_coeff_quadrilateral(int p){
    return (p + 1) * (p + 1);
}

KOKKOS_INLINE_FUNCTION
void get_face_pts_order_wrt_orient0_quadrilateral(const int orient, const int npts,
        Kokkos::View<int*> pts_order) {

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


/* --------------------------------------
        Hexahedron Shape Definitions
----------------------------------------*/
inline
int get_num_basis_coeff_hexahedron(int p){
    return (p + 1) * (p + 1) * (p + 1);
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

KOKKOS_INLINE_FUNCTION
void get_face_pts_order_wrt_orient0_hexahedron(const int orient, const int npts,
        Kokkos::View<int*> pts_order) {

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

inline
Shape::Shape(ShapeType shape_type){
    if (shape_type == ShapeType::Segment){
        // set methods
        get_num_basis_coeff = get_num_basis_coeff_segment;
        get_quadrature_order_pointer =
            SegmentQuadrature::get_gausslegendre_quadrature_order;
        get_quadrature_data =
            SegmentQuadrature::get_quadrature_gauss_legendre;
        
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

        // set constants
        NDIMS = 2;
        NFACES = 4;
        NCORNERS = 4;
        NUM_ORIENT_PER_FACE = 2;
        name = "Quadrilateral";
        type = ShapeType::Quadrilateral;
    }
    if (shape_type == ShapeType::Hexahedron){
        // set methods
        get_num_basis_coeff = get_num_basis_coeff_hexahedron;
        get_quadrature_order_pointer =
            QuadratureTools::get_gausslegendre_quadrature_order;
        get_quadrature_data =
            HexahedronQuadrature::get_quadrature_gauss_legendre;
        get_points_on_face = get_points_on_face_hexahedron;

        // set constants
        NDIMS = 3;
        NCORNERS = 8;
        NFACES = 6;
        NUM_ORIENT_PER_FACE = 8;
        name = "Hexahedron";
        type = ShapeType::Hexahedron;

    }
}

KOKKOS_INLINE_FUNCTION
void Shape::get_face_pts_order_wrt_orient0(const int orient, const int npts,
        Kokkos::View<int*> pts_order) const {
    if (type == ShapeType::Quadrilateral){
        get_face_pts_order_wrt_orient0_quadrilateral(orient, npts, pts_order);
    }
    else if (type == ShapeType::Hexahedron){
        get_face_pts_order_wrt_orient0_hexahedron(orient, npts, pts_order);    
    } else {
        printf("Not Implemented Error"); // TODO: Figure out GPU throws
    }
}

inline
int Shape::get_quadrature_order(const int order_) {
    return get_quadrature_order_pointer(order_, NDIMS);
}

} // end namespace Basis
