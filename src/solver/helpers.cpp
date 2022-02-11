// using view_type_1D = Kokkos::View<rtype*>;
// using host_view_type_1D = view_type_1D::HostMirror;

// using view_type_2D = Kokkos::View<rtype**>;
// using host_view_type_2D = view_type_2D::HostMirror;

namespace VolumeHelpers {



VolumeHelperFunctor::VolumeHelperFunctor(Basis::Basis basis){


    get_quadrature(basis, basis.get_order());
    // std::cout<<basis.get_order()<<std::endl;
    // testing = 2;
}

KOKKOS_FUNCTION
void VolumeHelperFunctor::operator()(const int ie) const {

	 // printf("Hello from ie = %i\n", testing);
    printf("Hello from quad_pts = %f\n", quad_pts(0, 0));
    printf("Hello from quad_wts = %f\n", quad_wts(0));

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

    host_view_type_2D h_quad_pts = Kokkos::create_mirror_view(quad_pts);
    host_view_type_1D h_quad_wts = Kokkos::create_mirror_view(quad_wts);

    basis.shape.get_quadrature_data(qorder, nq_1d, h_quad_pts, h_quad_wts);

    Kokkos::deep_copy(quad_pts, h_quad_pts);
    Kokkos::deep_copy(quad_wts, h_quad_wts);

}

} // end namespace VolumeHelper