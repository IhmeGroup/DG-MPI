#ifndef DG_LINEAR_ALGEBRA_H
#define DG_LINEAR_ALGEBRA_H

#include "common/defines.h"

#include<Kokkos_Core.hpp>
#include "KokkosBatched_LU_Decl.hpp"
#include "KokkosBatched_LU_Serial_Impl.hpp"
#include "KokkosBatched_LU_Team_Impl.hpp"
#include "KokkosBatched_SolveLU_Decl.hpp"

#include "KokkosBatched_Gemm_Decl.hpp"
#include "KokkosBatched_Gemm_Serial_Impl.hpp"
#include "KokkosBatched_Gemm_Team_Impl.hpp"


using namespace KokkosBatched;

namespace Math {

    template<typename ViewType> KOKKOS_INLINE_FUNCTION
    void identity(ViewType mat);

    template<typename ViewType1, typename ViewType2> KOKKOS_INLINE_FUNCTION
    void invA(const ViewType1 mat, ViewType2 imat);

    template<typename ViewType1, typename ViewType2> KOKKOS_INLINE_FUNCTION
    void cAxBT_to_C(rtype c, const ViewType1& A,
        const ViewType2& B, ViewType2& C);

    template<typename ViewType1, typename ViewType2, typename ViewType3> KOKKOS_INLINE_FUNCTION
    void cATxB_to_C(rtype c, const ViewType1& A,
        const ViewType2& B, ViewType3& C);

    template<typename ViewType1, typename ViewType2, typename ViewType3> KOKKOS_INLINE_FUNCTION
    void cAxB_to_C(rtype c, const ViewType1& A,
        const ViewType2& B, ViewType3& C);

    template<typename ViewType> KOKKOS_INLINE_FUNCTION
    void det(const ViewType &mat, rtype &det);
}

#include "math/linear_algebra.cpp"

#endif // DG_LINEAR_ALGEBRA_H
