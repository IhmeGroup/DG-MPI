#include "numerics/basis/basis.h"
#include "numerics/basis/tools.h"

namespace Basis {

/* --------------------------------------
    LagrangeSeg Method Definitions
----------------------------------------*/
void get_values_lagrangeseg(host_view_type_2D_ls quad_pts,
        host_view_type_2D_ls basis_val, const int order,
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
            BasisTools::get_lagrange_basis_val_1D(quad_pts(iq, 0), xnodes,
                Kokkos::subview(basis_val, iq, Kokkos::ALL()));
        }
    }

}

void get_grads_lagrangeseg(host_view_type_2D_ls quad_pts,
        host_view_type_3D_ls basis_ref_grad, const int order,
        void (*get_1d_nodes)(rtype start, rtype stop, int nnodes,
        host_view_type_1D &xnodes)){

    int nq = quad_pts.extent(0);

    if (order > 0){
        host_view_type_1D xnodes("xnodes", order + 1);
        get_1d_nodes(-1., 1., order + 1, xnodes);
        for (int iq = 0; iq < nq; iq++){
            BasisTools::get_lagrange_basis_grad_1D(quad_pts(iq, 0), xnodes,
                Kokkos::subview(basis_ref_grad, iq, Kokkos::ALL(), 0));
        }
    }
}

/* --------------------------------------
    LagrangeQuad Method Definitions
----------------------------------------*/
void get_values_lagrangequad(host_view_type_2D_ls quad_pts,
        host_view_type_2D_ls basis_val, const int order,
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

void get_grads_lagrangequad(host_view_type_2D_ls quad_pts,
        host_view_type_3D_ls basis_ref_grad, const int order,
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
void get_values_lagrangehex(host_view_type_2D_ls quad_pts,
        host_view_type_2D_ls basis_val, const int order,
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

void get_grads_lagrangehex(host_view_type_2D_ls quad_pts,
        host_view_type_3D_ls basis_ref_grad, const int order,
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
void get_values_legendreseg(host_view_type_2D_ls quad_pts,
        host_view_type_2D_ls basis_val, const int order,
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

void get_grads_legendreseg(host_view_type_2D_ls quad_pts,
        host_view_type_3D_ls basis_ref_grad, const int order,
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
void get_values_legendrequad(host_view_type_2D_ls quad_pts,
        host_view_type_2D_ls basis_val, const int order,
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


void get_grads_legendrequad(host_view_type_2D_ls quad_pts,
        host_view_type_3D_ls basis_ref_grad, const int order,
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
void get_values_legendrehex(host_view_type_2D_ls quad_pts,
        host_view_type_2D_ls basis_val, const int order,
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


void get_grads_legendrehex(host_view_type_2D_ls quad_pts,
        host_view_type_3D_ls basis_ref_grad, const int order,
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
        // Note: No face_shape as 1D is not supported
    }
    if (basis_type == BasisType::LagrangeQuad){
        get_values_pointer = get_values_lagrangequad;
        get_grads_pointer = get_grads_lagrangequad;
        name = "LagrangeQuad";
        shape = Shape(enum_from_string<ShapeType>("Quadrilateral"));
        face_shape = Shape(enum_from_string<ShapeType>("Segment"));
    }
    if (basis_type == BasisType::LagrangeHex){
        get_values_pointer = get_values_lagrangehex;
        get_grads_pointer = get_grads_lagrangehex;
        name = "LagrangeHex";
        shape = Shape(enum_from_string<ShapeType>("Hexahedron"));
        face_shape = Shape(enum_from_string<ShapeType>("Quadrilateral"));

    }
    if (basis_type == BasisType::LegendreSeg){
        get_values_pointer = get_values_legendreseg;
        get_grads_pointer = get_grads_legendreseg;
        name = "LegendreSeg";
        shape = Shape(enum_from_string<ShapeType>("Segment"));
        // Note: No face_shape as 1D is not supported
    }
    if (basis_type == BasisType::LegendreQuad){
        get_values_pointer = get_values_legendrequad;
        get_grads_pointer = get_grads_legendrequad;
        name = "LegendreQuad";
        shape = Shape(enum_from_string<ShapeType>("Quadrilateral"));
        face_shape = Shape(enum_from_string<ShapeType>("Segment"));

    }
    if (basis_type == BasisType::LegendreHex){
        get_values_pointer = get_values_legendrehex;
        get_grads_pointer = get_grads_legendrehex;
        name = "LegendreHex";
        shape = Shape(enum_from_string<ShapeType>("Hexahedron"));
        face_shape = Shape(enum_from_string<ShapeType>("Quadrilateral"));

    }

    this->nb = shape.get_num_basis_coeff(order);

}

void Basis::get_values(host_view_type_2D_ls quad_pts,
    host_view_type_2D_ls basis_val) {

    get_values_pointer(quad_pts, basis_val,
        order, get_1d_nodes);
}

void Basis::get_grads(host_view_type_2D_ls quad_pts,
    host_view_type_3D_ls basis_ref_grad) {

    get_grads_pointer(quad_pts, basis_ref_grad,
        order, get_1d_nodes);
}

view_type_3D Basis::get_face_basis_ref_grad_for_normals(const int gorder,
            const host_view_type_3D h_quad_pts) {

    if (shape.get_face_type() == ShapeType::Segment) {
        Basis face_gbasis(BasisType::LagrangeSeg, gorder);        
        int gnb = face_gbasis.shape.get_num_basis_coeff(face_gbasis.get_order());
        int nqf = (int)h_quad_pts.extent(1);
        view_type_3D face_gbasis_ref_grad("face gbasis ref grad", nqf, gnb, 1);
        host_view_type_3D h_face_gbasis_ref_grad = Kokkos::create_mirror_view(face_gbasis_ref_grad);

        // extract reference quadrature for a face from quad_pts
        auto ref_quad_pts = Kokkos::subview(h_quad_pts, 0, Kokkos::ALL, Kokkos::make_pair((unsigned)0, (unsigned)1));
        face_gbasis.get_grads(ref_quad_pts, h_face_gbasis_ref_grad);

        Kokkos::deep_copy(face_gbasis_ref_grad, h_face_gbasis_ref_grad);

        return face_gbasis_ref_grad;
    }
    else if (shape.get_face_type() == ShapeType::Quadrilateral) {
        Basis face_gbasis(BasisType::LagrangeQuad, gorder);
        int gnb = face_gbasis.shape.get_num_basis_coeff(face_gbasis.get_order());
        int nqf = (int)h_quad_pts.extent(1);
        view_type_3D face_gbasis_ref_grad("face gbasis ref grad", nqf, gnb, 2);
        host_view_type_3D h_face_gbasis_ref_grad = Kokkos::create_mirror_view(face_gbasis_ref_grad);

        // extract reference quadrature for a face from quad_pts
        auto ref_quad_pts = Kokkos::subview(h_quad_pts, 0, Kokkos::ALL, Kokkos::make_pair((unsigned)0, (unsigned)2));
        face_gbasis.get_grads(ref_quad_pts, h_face_gbasis_ref_grad);

        Kokkos::deep_copy(face_gbasis_ref_grad, h_face_gbasis_ref_grad);      

        return face_gbasis_ref_grad;
    }


}



} // end namespace Basis
