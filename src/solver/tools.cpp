#include "math/linear_algebra.h"


namespace SolverTools {

inline
void calculate_volume_flux_integral(const int num_elems, view_type_3D basis_ref_grad,
    view_type_4D F_quad, view_type_3D res){

    // unpack loop constants
    const int nq = basis_ref_grad.extent(0);
    const int nb = basis_ref_grad.extent(1);
    const int ns = F_quad.extent(2);
    const int ndims = basis_ref_grad.extent(2);

    // TODO: Use batched kernels + teampolicy to optimize this!
    // Kokkos::parallel_for("calc vol flux integral", Kokkos::TeamPolicy<>( num_elems,
    //     Kokkos::AUTO), KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& member){
    Kokkos::parallel_for(num_elems, KOKKOS_LAMBDA(const int& elem_ID){
        // const int elem_ID = member.league_rank();
        
        // TODO: Optimize this loop either with batched Kokkos kernels or 
        // with hierarchical parralelism
        for (int ib = 0; ib < nb; ib++){
            for (int is = 0; is < ns; is++){
                for (int iq = 0; iq < nq; iq++){
                    for (int id = 0; id < ndims; id++){
                        res(elem_ID, ib, is) += basis_ref_grad(iq, ib, id) *
                            F_quad(elem_ID, iq, is, id);
                    }
                }
            }
        }
    });
}

inline
void calculate_face_flux_integral(const int num_elems, view_type_2D basis_val, 
    view_type_3D F_quad, view_type_3D res){

    Kokkos::parallel_for("face flux integral", Kokkos::TeamPolicy<>( num_elems,
        Kokkos::AUTO), KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& member){
        const int elem_ID = member.league_rank();
        auto F_elem = Kokkos::subview(F_quad, elem_ID, Kokkos::ALL(), Kokkos::ALL());
        auto res_elem = Kokkos::subview(res, elem_ID, Kokkos::ALL(), Kokkos::ALL());
        
        // if (member.team_rank()==0){
        // // for (unsigned i = 0 ; i < F_elem.extent(0); i++){
        // //     for (unsigned j = 0; j < F_elem.extent(1); j++){
        // //         printf("F_elem(%i, %i, %i)=%f\n", member.league_rank(), i, j, F_elem(i, j));
        // //     }
        // // }

        // for (unsigned i = 0 ; i < basis_val.extent(0); i++){
        //     for (unsigned j = 0; j < basis_val.extent(1); j++){
        //         printf("basis_val(%i, %i)=%f\n", i, j, basis_val(i, j));
        //     }
        // }
        // }
        Math::cATxBpC_to_C(1., basis_val, F_elem, res_elem, member);
        member.team_barrier();
    });
}


template<typename ViewType_iMM, typename ViewType2D, typename ViewType1D_djac, 
typename ViewType1D_quadwts, typename ViewType2D_f, typename ViewType2D_state>
KOKKOS_INLINE_FUNCTION
void L2_projection(ViewType_iMM iMM, ViewType2D basis_val, ViewType1D_djac djac,
    ViewType1D_quadwts quad_wts, ViewType2D_f f, ViewType2D_state U,
    const membertype& member){

    const int nq = basis_val.extent(0);
    const int nb = basis_val.extent(1); 
    const int ns = f.extent(1);

    // scratch memory for rhs
    scratch_view_2D_rtype rhs(member.team_scratch( 1 ), nb, ns);

    // Multiply quad_wts * djac * f
    Kokkos::parallel_for(Kokkos::TeamThreadRange(member, nq), KOKKOS_LAMBDA ( const int iq ) {
        auto f_iq = Kokkos::subview(f, iq, Kokkos::ALL());
        Math::cA_to_A(quad_wts(iq) * djac(iq), f_iq);
    });
    member.team_barrier();

    // for (int i = 0; i < nq; i++){
    //     printf("djac(%i, %i)=%f\n", member.league_rank(), i, djac(i));
    // }


    // for (int i = 0; i < nq; i++){
    //     printf("quad_wts(%i, %i)=%f\n", member.league_rank(), i, quad_wts(i));
    // }

    Math::cATxB_to_C(1., basis_val, f, rhs, member);
    member.team_barrier();

    // for (int i = 0; i < nb; i++){
    //     for (int j = 0; j < ns; j++)
    //     {
    //         printf("rhs(%i, %i, %i)=%f\n", member.league_rank(), i, j, rhs(i, j));
    //     }
    // }

    Math::cAxB_to_C(1., iMM, rhs, U, member);
    member.team_barrier();


    // for (int i = 0; i < nb; i++){
    //     for (int j = 0; j < ns; j++)
    //     {
    //         printf("rhs(%i, %i, %i)=%f\n", member.league_rank(), i, j, rhs(i, j));
    //     }
    // }
    // for (int is = 0; is < 4; is++){
    //     for (int ib = 0; ib < 4; ib++){
    //         printf("U(%i, %i, %i)=%f\n", member.league_rank(),  ib, is, U(ib, is));
    //     }
    // }
    member.team_barrier();
}

template<typename ViewType_iMM, typename ViewType_res> inline
void mult_inv_mass_matrix(const rtype dt, const ViewType_iMM iMM_elems,
    const ViewType_res res, ViewType_res dU){

    Kokkos::parallel_for("mult inv mass matrix", Kokkos::TeamPolicy<>( (int)res.extent(0),
        Kokkos::AUTO), KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& member){
        const int elem_ID = member.league_rank();

        auto iMM = Kokkos::subview(iMM_elems, elem_ID, Kokkos::ALL(), Kokkos::ALL());
        auto res_elem = Kokkos::subview(res, elem_ID, Kokkos::ALL(), Kokkos::ALL());
        auto dU_elem = Kokkos::subview(dU, elem_ID, Kokkos::ALL(), Kokkos::ALL());
        Math::cAxB_to_C(dt, iMM, res_elem, dU_elem, member);
        member.team_barrier();
    });

}

} // end namespace SolverTools