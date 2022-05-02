#ifndef DG_LINEAR_ALGEBRA_H
#define DG_LINEAR_ALGEBRA_H

#include "common/defines.h"

#include<Kokkos_Core.hpp>

#include <KokkosBatched_Scale_Decl.hpp>
#include <KokkosBatched_Scale_Impl.hpp>

#include "KokkosBatched_Copy_Decl.hpp"
#include "KokkosBatched_Copy_Impl.hpp"

#include "KokkosBatched_LU_Decl.hpp"
#include "KokkosBatched_LU_Serial_Impl.hpp"
#include "KokkosBatched_LU_Team_Impl.hpp"
#include "KokkosBatched_SolveLU_Decl.hpp"

#include "KokkosBatched_Gemm_Decl.hpp"
#include "KokkosBatched_Gemm_Serial_Impl.hpp"
#include "KokkosBatched_Gemm_Team_Impl.hpp"

#include "KokkosBlas1_axpby.hpp"
#include "KokkosBlas1_fill.hpp"
#include "KokkosBlas1_update.hpp"


using namespace KokkosBatched;

/*
namespace Math
--------------

The following namespace provides wrappers around various math operations. 
Each function utilizes Kokkos kernels to conduct various mathematical 
operations. (Example: computing the inverse of matrix A)

Some things to note:

    1. We use function overloading to access the serial vs team versions 
       of the Kokkos batched functions. For the team versions, we must pass
       the member of the TeamPolicy.
    2. These are currently only batched Kokkos kernels wrappers, meaning
       each time they are called they MUST be called inside a Kokkos 
       parallel_for

    TODO: 
       Add overloaded kernels for large gemms
       Add additional math tests
*/
namespace Math {
    
    /*
    Conduct a dot product

    Outputs:
    --------
        returns an rtype
    */
    template<unsigned N> KOKKOS_INLINE_FUNCTION
    rtype dot(const rtype* a, const rtype* b);

    /*
    Conduct a cross product (3 x 3 only)

    Outputs:
    --------
        returns a 1D view
    */
    template<typename ViewType1D> KOKKOS_INLINE_FUNCTION
    void cross(const rtype* a, const rtype* b, ViewType1D c);

    /*
    Construct an identity matrix

    Outputs:
    --------
        mat: identity matrix sized by the extent of the provided View
    */
    template<typename ViewType> KOKKOS_INLINE_FUNCTION
    void identity(ViewType mat);

    /*
    Scales a matrix A by constant c

    Inputs:
    -------
        c: scalar value
        A: matrix to be scaled
    Outputs:
    --------
        A: returned A matrix is scaled by c
    */
    template<typename ScalarType, typename ViewType> KOKKOS_INLINE_FUNCTION
    void cA_to_A(const ScalarType c, ViewType A);
    
    /* Team scaling of matrix A by constant c */
    template<typename MemberType, typename ScalarType, typename ViewType> KOKKOS_INLINE_FUNCTION
    void cA_to_A(const ScalarType c, ViewType A, const MemberType& member);
    
    /*
    Fills a view with the prescribed value

    Inputs:
    -------
        num_entries: number of entries in a view
        A: view to be set
        c: value view is set to

    Outputs:
    --------
        A: set view
    */
    inline
    void fill(const int num_entries, rtype* A, rtype c);

    /* 
    TODO: Implement serial copy of A to B
    */

    /*
    Team copy of A to B
    This function takes the View A and directly copies it into View B
    */
    template<typename MemberType, typename ViewType1, typename ViewType2> KOKKOS_INLINE_FUNCTION
    void copy_A_to_B(const ViewType1& A, ViewType2& B, const MemberType& member);

    /*
    Computes B = B + cA
    
    NOTE: This is not implemented with batched kokkos it must be called on the host
    */
    inline
    void cApB_to_B(const unsigned nA, const rtype c, rtype* A, rtype* B);

    /*
    Computes C = B + cA
    
    NOTE: This is not implemented with batched kokkos it must be called on the host
    */
    inline
    void cApB_to_C(const unsigned nA, const rtype c, const rtype* A, const rtype* B, rtype* C);

    /*
    Computes the inverse of the matrix A

    Inputs:
    -------
        mat: matrix to be inverted
    Outputs:
    --------
        imat: inverted matrix
    */
    template<typename ViewType1, typename ViewType2> KOKKOS_INLINE_FUNCTION
    void invA(const ViewType1 mat, ViewType2 imat);

    /*
    Team version of computing the inverse of A
    */
    template<typename MemberType, typename ViewType1, typename ViewType2> KOKKOS_INLINE_FUNCTION
    void invA(const ViewType1 mat, ViewType2 imat, const MemberType& member);

    /*
    Computes cAB^T and stores in C

    Inputs:
    -------
        c: scalar value
        A: matrix shape [n x m]
        B: matrix shape [p x m] (note the transpose is taken)

    Outputs:
    --------
        C: matrix shape [n x p]
    */
    template<typename ViewType1, typename ViewType2> KOKKOS_INLINE_FUNCTION
    void cAxBT_to_C(rtype c, const ViewType1& A,
        const ViewType2& B, ViewType2& C);

    /*
    Team version of cAxBT_to_C
    */
    template<typename ViewTypeA, typename ViewTypeB, 
    typename ViewTypeC, typename MemberType> KOKKOS_INLINE_FUNCTION
    void cAxBT_to_C(rtype c, const ViewTypeA& A,
        const ViewTypeB& B, ViewTypeC& C, const MemberType& member);

    /*
    Computes cA^T*B and stores in C

    Inputs:
    -------
        c: scalar value
        A: matrix shape [n x m]
        B: matrix shape [n x p]

    Outputs:
    --------
        C: matrix shape [m x p]
    */
    template<typename ViewType1, typename ViewType2, 
    typename ViewType3> KOKKOS_INLINE_FUNCTION
    void cATxB_to_C(rtype c, const ViewType1& A,
        const ViewType2& B, ViewType3& C);

    /*
    Team version of cA^T*B -> C
    */
    template<typename MemberType, typename ViewType1, typename ViewType2, 
    typename ViewType3> KOKKOS_INLINE_FUNCTION
    void cATxB_to_C(rtype c, const ViewType1& A, const ViewType2& B, 
        ViewType3& C, const MemberType& member);

    /*
    Computes cA^T*B + C and stores in C

    Inputs:
    -------
        c: scalar value
        A: matrix shape [n x m]
        B: matrix shape [n x p]

    Outputs:
    --------
        C: matrix shape [m x p]
    */
    template<typename ViewType1, typename ViewType2, 
    typename ViewType3> KOKKOS_INLINE_FUNCTION
    void cATxBpC_to_C(rtype c, const ViewType1& A, const ViewType2& B, ViewType3& C);

    /*
    Team version of cA^T*B + C -> C
    */
    template<typename MemberType, typename ViewType1, typename ViewType2, 
    typename ViewType3> KOKKOS_INLINE_FUNCTION
    void cATxBpC_to_C(rtype c, const ViewType1& A, const ViewType2& B, ViewType3& C, 
        const MemberType& member);

    /*
    Computes cAB and stores in C

    Inputs:
    -------
        c: scalar
        A: matrix shape [n x m]
        B: matrix shape [m x p]

    Outputs:
    --------
        C: matrix shape [n x p]
    */
    template<typename ViewType1, typename ViewType2, typename ViewType3> KOKKOS_INLINE_FUNCTION
    void cAxB_to_C(rtype c, const ViewType1& A,
        const ViewType2& B, ViewType3& C);

    /*
    Team version of cAB -> C
    */
    template<typename MemberType, typename ViewType1, typename ViewType2, 
    typename ViewType3> KOKKOS_INLINE_FUNCTION
    void cAxB_to_C(rtype c, const ViewType1& A, const ViewType2& B, ViewType3& C, 
        const MemberType& member);

    /*
    Calculates the determinant of a matrix

    Note: Not general, only works with 1x1, 2x2, and 3x3 matrices

    Inputs:
    -------
        mat: matrix shape [n x n]

    Outputs:
    --------
        det: determinant of mat
    */
    template<typename ViewType> KOKKOS_INLINE_FUNCTION
    void det(const ViewType &mat, rtype &det);
}

#include "math/linear_algebra.cpp"

#endif // DG_LINEAR_ALGEBRA_H
