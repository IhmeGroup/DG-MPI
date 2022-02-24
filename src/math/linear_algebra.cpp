namespace Math {


    template<typename ViewType> KOKKOS_INLINE_FUNCTION
    void identity(ViewType mat){
        for (unsigned i = 0; i < mat.extent(0); i++){
            for (unsigned j = 0; j < mat.extent(1); j++){
                if ( i == j ) {mat(i, j) = 1.;}
            }
        }
    }

    template<typename ScalarType, typename ViewType> KOKKOS_INLINE_FUNCTION
    void cA_to_A(const ScalarType c, ViewType A){
        SerialScale::invoke(c, A);
    }


    template<typename MemberType, typename ViewType1, typename ViewType2> KOKKOS_INLINE_FUNCTION
    void team_copy_A_to_B(const ViewType1& A, ViewType2& B, const MemberType& member){
        TeamCopy<MemberType, Trans::NoTranspose>::invoke(member, A, B);
    }


    template<typename ViewType1, typename ViewType2> KOKKOS_INLINE_FUNCTION
    void invA(const ViewType1 mat, ViewType2 imat){

        assert(mat.extent(0) == mat.extent(1));
        assert(imat.extent(0) == imat.extent(1));
        // need to make imat identity prior to solve
        identity(imat);
        SerialLU<Algo::Level3::Unblocked>::invoke(mat);
        SerialSolveLU<Trans::NoTranspose, Algo::Level3::Unblocked>
            ::invoke(mat, imat);
    }

    template<typename MemberType, typename ViewType1, typename ViewType2> KOKKOS_INLINE_FUNCTION
    void team_invA(const ViewType1 mat, ViewType2 imat, const MemberType& member){

        assert(mat.extent(0) == mat.extent(1));
        assert(imat.extent(0) == imat.extent(1));
        // need to make imat identity prior to solve
        identity(imat);
        TeamLU<MemberType, Algo::Level3::Unblocked>::invoke(member, mat);
        TeamSolveLU<MemberType, Trans::NoTranspose, Algo::Level3::Unblocked>
            ::invoke(member, mat, imat);
    }

    template<typename ViewType1, typename ViewType2> KOKKOS_INLINE_FUNCTION
    void cAxBT_to_C(rtype c, const ViewType1& A, const ViewType2& B, ViewType2& C){

        SerialGemm<Trans::NoTranspose, Trans::Transpose, Algo::Gemm::Unblocked>
            ::invoke(c, A, B, 0., C);
    }

    template<typename ViewType1, typename ViewType2, typename ViewType3> KOKKOS_INLINE_FUNCTION
    void cATxB_to_C(rtype c, const ViewType1& A, const ViewType2& B, ViewType3& C){

        SerialGemm<Trans::Transpose, Trans::NoTranspose, Algo::Gemm::Unblocked>
            ::invoke(c, A, B, 0., C);
    }

    template<typename MemberType, typename ViewType1, typename ViewType2, typename ViewType3> KOKKOS_INLINE_FUNCTION
    void team_cATxB_to_C(rtype c, const ViewType1& A, const ViewType2& B, ViewType3& C, const MemberType& member){

        TeamGemm<MemberType, Trans::Transpose, Trans::NoTranspose, Algo::Gemm::Unblocked>
            ::invoke(member, c, A, B, 0., C);
    }

    template<typename ViewType1, typename ViewType2, typename ViewType3> KOKKOS_INLINE_FUNCTION
    void cAxB_to_C(rtype c, const ViewType1& A, const ViewType2& B, ViewType3& C){

        SerialGemm<Trans::NoTranspose, Trans::NoTranspose, Algo::Gemm::Unblocked>
            ::invoke(c, A, B, 0., C);
    }

    template<typename ViewType> KOKKOS_INLINE_FUNCTION
    void det(const ViewType &mat, rtype &det) {
        assert(mat.extent(0) == mat.extent(1));
        int dim = mat.extent(0);

        if (dim==1) {
            det=mat(0, 0);
        }
        if (dim==2) {
            det=mat(0, 0)*mat(1, 1)-mat(0, 1)*mat(1, 0);
        }
        if (dim==3) {
            det=mat(0, 0)*mat(1, 1)*mat(2, 2)-mat(0, 0)*mat(1, 2)*mat(2, 1)
                -mat(0, 1)*mat(1, 0)*mat(2, 2)+mat(0, 1)*mat(1, 2)*mat(2, 0)
                +mat(0, 2)*mat(1, 0)*mat(2, 1)-mat(0, 2)*mat(1, 1)*mat(2, 0);
        }
    }
}
