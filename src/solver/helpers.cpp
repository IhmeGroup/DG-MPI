using view_type_1D = Kokkos::View<rtype*>;
using host_view_type_1D = view_type_1D::HostMirror;

using view_type_2D = Kokkos::View<rtype**>;
using host_view_type_2D = view_type_2D::HostMirror;

namespace VolumeHelpers {



VolumeHelperFunctor::VolumeHelperFunctor(Basis::Basis basis){


    get_quadrature(basis, basis.get_order());
    // std::cout<<basis.get_order()<<std::endl;
    // testing = 2;
}

KOKKOS_FUNCTION
void VolumeHelperFunctor::operator()(const int ie) const {

	 // printf("Hello from ie = %i\n", testing);


}

void VolumeHelperFunctor::get_quadrature(
    Basis::Basis basis, const int order){

    int qorder = basis.shape.get_quadrature_order(order);
    int nq;

    // need to establish an initial size for views prior
    // to resizing inside of get_quadrature_data
    Kokkos::resize(quad_pts, 2, 1);
    Kokkos::resize(quad_wts, 1);

    host_view_type_2D h_quad_pts = Kokkos::create_mirror_view(quad_pts);
    host_view_type_1D h_quad_wts = Kokkos::create_mirror_view(quad_wts);
    // h_quad_pts(1, 0) = 4.;
    basis.shape.get_quadrature_data(qorder, nq, h_quad_pts, h_quad_wts);

    Kokkos::resize(quad_pts, h_quad_pts.extent(0), h_quad_pts.extent(1));
    Kokkos::resize(quad_wts, h_quad_wts.extent(0));
    
    std::cout<<h_quad_wts(0)<<std::endl;
    Kokkos::deep_copy(quad_pts, h_quad_pts);
    Kokkos::deep_copy(quad_wts, h_quad_wts);

}

} // end namespace VolumeHelper