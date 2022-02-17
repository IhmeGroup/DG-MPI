#ifndef DG_LINEAR_ALGEBRA_H
#define DG_LINEAR_ALGEBRA_H

#include "common/defines.h"

#include<Kokkos_Core.hpp>
#include "KokkosBatched_LU_Decl.hpp"
#include "KokkosBatched_LU_Serial_Impl.hpp"
#include "KokkosBatched_LU_Team_Impl.hpp"
#include "KokkosBatched_SolveLU_Decl.hpp"
// #include<KokkosBlas_gesv.hpp>

using namespace KokkosBatched;

namespace Math {

	template<typename ViewType> KOKKOS_INLINE_FUNCTION
	void invA(const ViewType& mat, ViewType& imat);

}

#include "math/linear_algebra.cpp"

#endif // DG_LINEAR_ALGEBRA_H