// #include "common/defines.h"
// #include "physics/base/data.h"
// #include "physics/base/functions.h"


// namespace BaseFcnType {

// template<int dim>
// Uniform::Uniform(Physics::PhysicsBase<dim> &physics, const rtype *state){

//     int NS = physics.get_NS();
//     for (int is = 0; is < NS; is++){
//         physics->state[is] = state[is];
//     }
// }

// template<int dim> DG_KOKKOS_FUNCTION
// void Uniform::get_state(Physics::PhysicsBase<dim> &physics, const rtype *x, const rtype *t,
//     Kokkos::View<rtype*> Uq){

//     int NS = physics.get_NS();
//     for (int is = 0; is < NS; is++){
//         Uq(is) = physics->state[is];
//     }
// }

// } // end namespace BaseFcnType


namespace BaseConvNumFluxType {


template<unsigned dim> KOKKOS_INLINE_FUNCTION
void compute_flux_laxfriedrichs(const Physics::Physics<dim>& physics,
    const rtype* UL, const rtype* UR, const rtype* N, 
    rtype* F, rtype* gUL, rtype* gUR){

        static constexpr int NS = physics.NUM_STATE_VARS;
        // allocate normalized normal vector
        rtype n_hat[dim];
        rtype n_mag = 0.0;
        for (unsigned idim = 0; idim < dim; idim++) {
            n_mag += N[idim] * N[idim];
        }
        n_mag = sqrt(n_mag);

        for (unsigned idim = 0; idim < dim; idim++) {
            n_hat[idim] = N[idim] / n_mag;
        }
        // allocate worker arrays
        rtype Fq[NS * dim]; // [NS, dim]
        rtype dU[NS];

        // Left flux
        rtype FqL[NS] = {0.0};
        physics.get_conv_flux_interior(UL, gUL, F, Fq);
        for (unsigned is = 0; is < NS; is++){
            for (unsigned idim = 0; idim < dim; idim++){
                FqL[is] += Fq[is + idim * NS] * n_hat[idim];
            }
        }
        // Right flux
        rtype FqR[NS] = {0.0};
        physics.get_conv_flux_interior(UR, gUR, F, Fq);
        for (unsigned is = 0; is < NS; is++){
            for (unsigned idim = 0; idim < dim; idim++){
                FqR[is] += Fq[is + idim * NS] * n_hat[idim];
            }
        }
        // Jump condition
        // dUq = UqR - UqL -> what the two lines below do
        for (unsigned is = 0; is < NS; is++){
            dU[is] = UR[is] - UL[is];
        }

        // Calculate the max wave speed
        rtype a = physics.get_maxwavespeed(UL);
        rtype aR = physics.get_maxwavespeed(UR);

        if (aR > a){
            a = aR;
        }

        // Put together -> n_mag * (0.5 *(FqL + FqR) - 0.5 * a * dUq)
        for (unsigned is = 0 ; is < NS; is++){
            F[is] = n_mag * (0.5 * (FqL[is] + FqR[is]) - 0.5 * a * dU[is]);
            // printf("F[%i]=%f\n", is, F[is]);
        }
}


// template <int dim> DG_KOKKOS_FUNCTION
// void PhysicsBase<dim>::get_conv_flux_projected(
//     Kokkos::View<const rtype*> Uq,
//     Kokkos::View<const rtype*> normals,
//     Kokkos::View<rtype*> Fproj){
//     // unpack
//     int NS = get_NS();

//     // allocate view for directional flux
//     Kokkos::View<rtype**> Fq("Fq", NS, dim);
//     conv_flux_physical(Uq, Fq);
//     KokkosBlas::gemv ("N", 1., Fq, normals, 1., Fproj);
// }


// template<int dim> DG_KOKKOS_FUNCTION
// void LaxFriedrichs::compute_flux(Physics::PhysicsBase<dim> &physics,
//             Kokkos::View<rtype*> UqL,
//             Kokkos::View<rtype*> UqR,
//             Kokkos::View<rtype*> normals,
//             Kokkos::View<rtype*> Fq){

//     // unpack
//     int NS = physics.get_NS();
//     // int dim = 2;
//     // Normalize the normal vectors
//     Kokkos::View<rtype*> n_hat("n_hat", dim);
//     rtype n_mag = KokkosBlas::nrm2(normals);
//     for (int i = 0; i < dim; i++){
//         n_hat(i) = normals(i)/n_mag;
//     }

//     // Left flux
//     Kokkos::View<rtype*> FqL("FqL", NS);
//     physics.get_conv_flux_projected(UqL, n_hat, FqL);

//     // Right flux
//     Kokkos::View<rtype*> FqR("FqR", NS);
//     physics.get_conv_flux_projected(UqR, n_hat, FqR);

//     // Jump condition
//     // dUq = UqR - UqL -> what the two lines below do
//     Kokkos::View<rtype*> dUq("dUq", NS);
//     Kokkos::deep_copy(dUq, UqR);
//     KokkosBlas::axpy(-1., UqL, dUq);

//     // Calculate the max wave speed
//     rtype a = physics.get_maxwavespeed(UqL);
//     rtype aR = physics.get_maxwavespeed(UqR);

//     if (aR > a){
//         a = aR;
//     }

//     // Put together -> n_mag * (0.5 *(FqL + FqR) - 0.5 * a * dUq)
//     KokkosBlas::axpby(0.5, FqL, 0.5, FqR);
//     auto Favg = FqR;
//     KokkosBlas::axpby(1., Favg, -0.5 * a, dUq);
//     KokkosBlas::scal(Fq, n_mag, dUq);
// };

// template void LaxFriedrichs::compute_flux<2>(Physics::PhysicsBase<2> &physics,
//             Kokkos::View<rtype*> UqL,
//             Kokkos::View<rtype*> UqR,
//             Kokkos::View<rtype*> normals,
//             Kokkos::View<rtype*> Fq);

// template void LaxFriedrichs::compute_flux<3>(Physics::PhysicsBase<3> &physics,
//             Kokkos::View<rtype*> UqL,
//             Kokkos::View<rtype*> UqR,
//             Kokkos::View<rtype*> normals,
//             Kokkos::View<rtype*> Fq);

} // end namespace BaseConvNumFluxType
