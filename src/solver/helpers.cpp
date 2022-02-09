
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
        Kokkos::resize(quad_wts, 2);

        basis.shape.get_quadrature_data(qorder, nq, quad_pts, quad_wts);

        std::cout<<quad_pts(1, 0)<<std::endl;
    }

} // end namespace VolumeHelper