// using view_type_1D = Kokkos::View<rtype*>;
// using host_view_type_1D = view_type_1D::HostMirror;

// using view_type_2D = Kokkos::View<rtype**>;
// using host_view_type_2D = view_type_2D::HostMirror;

namespace VolumeHelpers {



VolumeHelperFunctor::VolumeHelperFunctor(Basis::Basis basis){


    get_quadrature(basis, basis.get_order());
    get_reference_data(basis, basis.get_order());
    // std::cout<<basis.get_order()<<std::endl;
    // testing = 2;
}

KOKKOS_FUNCTION
void VolumeHelperFunctor::operator()(const int ie) const {


    printf("quad_wts: %f\n" , quad_wts(0));



}

void VolumeHelperFunctor::get_quadrature(
    Basis::Basis basis, const int order){

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
    Basis::Basis basis, const int order){
    

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



}

} // end namespace VolumeHelper