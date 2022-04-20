namespace Math {

    template<unsigned N> KOKKOS_INLINE_FUNCTION
    rtype dot(const rtype *a, const rtype *b){
        rtype res = 0.;
        for (unsigned i = 0; i < N; i++) {
            res += a[i]*b[i];
        }
        return res;
    }


    template<> KOKKOS_INLINE_FUNCTION
    rtype dot<2>(const rtype* a, const rtype* b) {
        return a[0]*b[0] + a[1]*b[1];
    }


    template<> KOKKOS_INLINE_FUNCTION
    rtype dot<3>(const rtype* a, const rtype* b) {
        return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
    }


    template<> KOKKOS_INLINE_FUNCTION
    rtype dot<4>(const rtype* a, const rtype* b) {
        return a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3];
    }


    template<> KOKKOS_INLINE_FUNCTION
    rtype dot<5>(const rtype* a, const rtype* b) {
        return a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3] + a[4]*b[4];
    }

    template<typename ViewType1D> KOKKOS_INLINE_FUNCTION
    void cross(const rtype* a, const rtype* b, ViewType1D c){
        c(0) = a[1] * b[2] - a[2] * b[1];
        c(1) = a[2] * b[0] - a[0] * b[2];
        c(2) = a[0] * b[1] - a[1] * b[0];
    }

    template<typename ViewType> KOKKOS_INLINE_FUNCTION
    void identity(ViewType mat){
        // TODO: this should be parrallelized
        for (unsigned long i = 0; i < mat.extent(0); i++){
            for (unsigned long j = 0; j < mat.extent(1); j++){
                if ( i == j ) {mat(i, j) = 1.;}
            }
        }
    }

    template<typename ScalarType, typename ViewType> KOKKOS_INLINE_FUNCTION
    void cA_to_A(const ScalarType c, ViewType A){
        SerialScale::invoke(c, A);
    }


    inline
    void fill(const int num_entries, rtype* A, rtype c){
        view_type_1D AA(A, num_entries);
        KokkosBlas::fill(AA, c);
    }

    template<typename MemberType, typename ViewType1, typename ViewType2> KOKKOS_INLINE_FUNCTION
    void copy_A_to_B(const ViewType1& A, ViewType2& B, const MemberType& member){
        TeamCopy<MemberType, Trans::NoTranspose>::invoke(member, A, B);
    }

    inline
    void cApB_to_B(const unsigned nA, const rtype c, rtype* A, rtype* B){
        Kokkos::View<rtype*> AA(A, nA);
        Kokkos::View<rtype*> BB(B, nA);
        KokkosBlas::axpy(c, AA, BB);

    }

    inline
    void cApB_to_C(const unsigned nA, const rtype c, const rtype* A, const rtype* B, rtype* C){
        Kokkos::View<const rtype*> AA(A, nA);
        Kokkos::View<const rtype*> BB(B, nA);
        Kokkos::View<rtype*> CC(C, nA);
        KokkosBlas::update(c, AA, 1.0, BB, 0.0, CC);

    }

    template<typename ViewType1, typename ViewType2> KOKKOS_INLINE_FUNCTION
    void invA(const ViewType1 mat, ViewType2 imat){
        // make sure that the matrices are square
        assert(mat.extent(0) == mat.extent(1));
        assert(imat.extent(0) == imat.extent(1));

        const long unsigned dim = mat.extent(0);
        if (dim == 1){
            rtype det = mat(0, 0);
            imat(0, 0) = 1./det;
        } else if (dim == 2){
            rtype det=mat(0, 0)*mat(1, 1)-mat(0, 1)*mat(1, 0);
            imat(0, 0)=mat(1, 1)/det;
            imat(0, 1)=-mat(0, 1)/det;
            imat(1, 0)=-mat(1, 0)/det;
            imat(1, 1)=mat(0, 0)/det;
        } else if (dim == 3){
            rtype det=mat(0, 0)*mat(1, 1)*mat(2, 2)-mat(0, 0)*mat(1, 2)*mat(2, 1)
                -mat(0, 1)*mat(1, 0)*mat(2, 2)+mat(0, 1)*mat(1, 2)*mat(2, 0)
                +mat(0, 2)*mat(1, 0)*mat(2, 1)-mat(0, 2)*mat(1, 1)*mat(2, 0);
            // det=1.0;
            imat(0, 0)=(mat(1, 1)*mat(2, 2)-mat(1, 2)*mat(2, 1))/det;
            imat(1, 0)=-(mat(1, 0)*mat(2, 2)-mat(1, 2)*mat(2, 0))/det;
            imat(2, 0)=(mat(1, 0)*mat(2, 1)-mat(1, 1)*mat(2, 0))/det;
            imat(0, 1)=-(mat(0, 1)*mat(2, 2)-mat(0, 2)*mat(2, 1))/det;
            imat(1, 1)=(mat(0, 0)*mat(2, 2)-mat(0, 2)*mat(2, 0))/det;
            imat(2, 1)=-(mat(0, 0)*mat(2, 1)-mat(0, 1)*mat(2, 0))/det;
            imat(0, 2)=(mat(0, 1)*mat(1, 2)-mat(0, 2)*mat(1, 1))/det;
            imat(1, 2)=-(mat(0, 0)*mat(1, 2)-mat(0, 2)*mat(1, 0))/det;
            imat(2, 2)=(mat(0, 0)*mat(1, 1)-mat(0, 1)*mat(1, 0))/det;

        } else {
        // need to make imat identity prior to solve
        identity(imat);
        // get the LU decomposition
        SerialLU<Algo::Level3::Unblocked>::invoke(mat);
        // solve the LU=I to make I=(UL)^-1
        SerialSolveLU<Trans::NoTranspose, Algo::Level3::Unblocked>
            ::invoke(mat, imat);

        }
    }


    template<typename MemberType, typename ViewType1, typename ViewType2> KOKKOS_INLINE_FUNCTION
    void invA(const ViewType1 mat, ViewType2 imat, const MemberType& member){
        // make sure that the matrices are square
        assert(mat.extent(0) == mat.extent(1));
        assert(imat.extent(0) == imat.extent(1));

        const long unsigned dim = mat.extent(0);
        if (dim == 1){
            rtype det = mat(0, 0);
            imat(0, 0) = 1./det;
        } else if (dim == 2){
            rtype det=mat(0, 0)*mat(1, 1)-mat(0, 1)*mat(1, 0);
            imat(0, 0)=mat(1, 1)/det;
            imat(0, 1)=-mat(0, 1)/det;
            imat(1, 0)=-mat(1, 0)/det;
            imat(1, 1)=mat(0, 0)/det;
        } else if (dim == 3){
            rtype det=mat(0, 0)*mat(1, 1)*mat(2, 2)-mat(0, 0)*mat(1, 2)*mat(2, 1)
                -mat(0, 1)*mat(1, 0)*mat(2, 2)+mat(0, 1)*mat(1, 2)*mat(2, 0)
                +mat(0, 2)*mat(1, 0)*mat(2, 1)-mat(0, 2)*mat(1, 1)*mat(2, 0);
            // det=1.0;
            imat(0, 0)=(mat(1, 1)*mat(2, 2)-mat(1, 2)*mat(2, 1))/det;
            imat(1, 0)=-(mat(1, 0)*mat(2, 2)-mat(1, 2)*mat(2, 0))/det;
            imat(2, 0)=(mat(1, 0)*mat(2, 1)-mat(1, 1)*mat(2, 0))/det;
            imat(0, 1)=-(mat(0, 1)*mat(2, 2)-mat(0, 2)*mat(2, 1))/det;
            imat(1, 1)=(mat(0, 0)*mat(2, 2)-mat(0, 2)*mat(2, 0))/det;
            imat(2, 1)=-(mat(0, 0)*mat(2, 1)-mat(0, 1)*mat(2, 0))/det;
            imat(0, 2)=(mat(0, 1)*mat(1, 2)-mat(0, 2)*mat(1, 1))/det;
            imat(1, 2)=-(mat(0, 0)*mat(1, 2)-mat(0, 2)*mat(1, 0))/det;
            imat(2, 2)=(mat(0, 0)*mat(1, 1)-mat(0, 1)*mat(1, 0))/det;

        } else {
            // need to make imat identity prior to solve
            identity(imat);
            // get the LU decomposition
            TeamLU<MemberType, Algo::Level3::Unblocked>::invoke(member, mat);
            // solve the LU=I to make I=(UL)^-1
            TeamSolveLU<MemberType, Trans::NoTranspose, Algo::Level3::Unblocked>
                ::invoke(member, mat, imat);
        }
    }

    template<typename ViewType1, typename ViewType2> KOKKOS_INLINE_FUNCTION
    void cAxBT_to_C(rtype c, const ViewType1& A, const ViewType2& B, ViewType2& C){
        
        SerialGemm<Trans::NoTranspose, Trans::Transpose, Algo::Gemm::Unblocked>
            ::invoke(c, A, B, 0., C);
    }

    template<typename ViewTypeA, typename ViewTypeB, typename ViewTypeC, typename MemberType> KOKKOS_INLINE_FUNCTION
    void cAxBT_to_C(rtype c, const ViewTypeA& A,
        const ViewTypeB& B, ViewTypeC& C, const MemberType& member){
        
        TeamGemm<MemberType, Trans::NoTranspose, Trans::Transpose, Algo::Gemm::Unblocked>
            ::invoke(member, c, A, B, 0., C);
        }


    template<typename ViewType1, typename ViewType2, typename ViewType3> KOKKOS_INLINE_FUNCTION
    void cATxB_to_C(rtype c, const ViewType1& A, const ViewType2& B, ViewType3& C){
        
        SerialGemm<Trans::Transpose, Trans::NoTranspose, Algo::Gemm::Unblocked>
            ::invoke(c, A, B, 0., C);
    }


    template<typename MemberType, typename ViewType1, typename ViewType2, 
    typename ViewType3> KOKKOS_INLINE_FUNCTION
    void cATxB_to_C(rtype c, const ViewType1& A, const ViewType2& B, ViewType3& C, 
        const MemberType& member){
        
        TeamGemm<MemberType, Trans::Transpose, Trans::NoTranspose, Algo::Gemm::Unblocked>
            ::invoke(member, c, A, B, 0., C);
    }

    template<typename ViewType1, typename ViewType2, typename ViewType3> KOKKOS_INLINE_FUNCTION
    void cATxBpC_to_C(rtype c, const ViewType1& A, const ViewType2& B, ViewType3& C){
        
        SerialGemm<Trans::Transpose, Trans::NoTranspose, Algo::Gemm::Unblocked>
            ::invoke(c, A, B, 1., C);
    }


    template<typename MemberType, typename ViewType1, typename ViewType2, 
    typename ViewType3> KOKKOS_INLINE_FUNCTION
    void cATxBpC_to_C(rtype c, const ViewType1& A, const ViewType2& B, ViewType3& C, 
        const MemberType& member){
        
        TeamGemm<MemberType, Trans::Transpose, Trans::NoTranspose, Algo::Gemm::Unblocked>
            ::invoke(member, c, A, B, 1., C);
    }


    template<typename ViewType1, typename ViewType2, typename ViewType3> KOKKOS_INLINE_FUNCTION
    void cAxB_to_C(rtype c, const ViewType1& A, const ViewType2& B, ViewType3& C){

        SerialGemm<Trans::NoTranspose, Trans::NoTranspose, Algo::Gemm::Unblocked>
            ::invoke(c, A, B, 0., C);
    }

    template<typename MemberType, typename ViewType1, typename ViewType2, typename ViewType3> KOKKOS_INLINE_FUNCTION
    void cAxB_to_C(rtype c, const ViewType1& A, const ViewType2& B, ViewType3& C, const MemberType& member){

        TeamGemm<MemberType, Trans::NoTranspose, Trans::NoTranspose, Algo::Gemm::Unblocked>
            ::invoke(member, c, A, B, 0., C);
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
