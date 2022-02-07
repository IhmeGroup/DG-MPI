namespace VolumeHelpers {

	VolumeHelperFunctor::VolumeHelperFunctor(){};

	KOKKOS_INLINE_FUNCTION
    void VolumeHelperFunctor::operator()(const int ie) const {

    	 printf("Hello from ie = %i\n", ie);
    }

} // end namespace VolumeHelper