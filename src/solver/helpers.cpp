
namespace VolumeHelpers {

    VolumeHelperFunctor::VolumeHelperFunctor(){

        

        testing = 2;
    }

	KOKKOS_FUNCTION
    void VolumeHelperFunctor::operator()(const int ie) const {

    	 // printf("Hello from ie = %i\n", testing);


    }

} // end namespace VolumeHelper