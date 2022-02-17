namespace Math {

	template<typename ViewType> KOKKOS_INLINE_FUNCTION
	void invA(const ViewType& mat, ViewType& imat){

		assert(mat.extent(0) == mat.extent(1));
		SerialLU<Algo::Level3::Unblocked>::invoke(mat);
		SerialSolveLU<Trans::NoTranspose, Algo::Level3::Blocked>
		::invoke(mat, imat);
	}

}