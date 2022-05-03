#ifndef DG_SOLVER_FLUX_FUNCTORS_H
#define DG_SOLVER_FLUX_FUNCTORS_H

#include "common/defines.h"
#include "physics/base/base.h"
#include "mesh/mesh.h"
#include <Kokkos_Core.hpp>


namespace FluxFunctors {


template<unsigned dim>
struct VolumeFluxesFunctor {

  public:
    VolumeFluxesFunctor(const Physics::Physics<dim> physics,
        view_type_3D Uq, view_type_4D vgUq, view_type_2D djac_elems, 
        view_type_4D ijac_elems, view_type_1D quad_wts);
    
    KOKKOS_INLINE_FUNCTION
    void operator()(const int ie, const int iq) const;

private:
    // set as compile time constant -> see common/defines.h
    static constexpr int NS = GLOBAL_NUM_SPECIES + 1 + dim;

    const Physics::Physics<dim> physics;

    view_type_3D Uq;
    view_type_4D vgUq;
    view_type_2D djac;
    view_type_4D ijac; 
    view_type_1D quad_wts;
    // bool need_state_grad;
};

template<unsigned dim>
VolumeFluxesFunctor<dim>::VolumeFluxesFunctor(const Physics::Physics<dim> physics,
    view_type_3D Uq, view_type_4D vgUq, view_type_2D djac_elems,
    view_type_4D ijac_elems,
    view_type_1D quad_wts) : physics{physics}, Uq{Uq}, vgUq{vgUq}, djac{djac_elems}, ijac{ijac_elems}, quad_wts{quad_wts} {

}

template<unsigned dim> KOKKOS_INLINE_FUNCTION
void VolumeFluxesFunctor<dim>::operator()(
    const int ie,
    const int iq) const {

    // local storage at a specific point
    rtype U[NS];
    rtype gU1[dim * NS];
    rtype gU2[dim * NS];
    rtype F[NS] = {0}; // to silence warnings TODO: Does this silence any warnings for us?
    rtype gF1[dim * NS];
    rtype gF2[dim * NS];
    unsigned idx;

    // read
    for (unsigned is = 0; is < NS; is++) {
        U[is] = Uq(ie, iq, is);
    }

    // TODO: Add state gradient capability for Navier-Stokes
    // if (need_state_grad) {
    //     // state gradient in reference space
    //     idx = 0;
    //     for (unsigned idim = 0; idim < dim; idim++) {
    //         for (unsigned is = 0; is < NS; is++) {
    //             gU2[idx] = vgUq(idim, iq, is, ie);
    //             idx++;
    //         }
    //     }
    //     // transform to physical space
    //     for (unsigned i = 0; i < dim * NS; i++) {
    //         gU1[i] = 0.;
    //     }
    //     for (unsigned id1 = 0; id1 < dim; id1++) { // reference dimension
    //         for (unsigned id2 = 0; id2 < dim; id2++) { // physical dimension
    //             for (unsigned is = 0; is < NS; is++) {
    //                 gU1[id2 * NS + is] += gU2[id1 * NS + is] * vijac(iq, id1, id2, ie);
    //             }
    //         }
    //     }
    // }

    // compute fluxes
    physics.get_conv_flux_interior(U, gU1, F, gF1);

    // "transform the flux back to reference space and prepare for integration"
    for (unsigned i = 0; i < dim * NS; i++) {
        gF2[i] = 0.;
    }
    for (unsigned id1 = 0; id1 < dim; id1++) { // reference dimension
        for (unsigned id2 = 0; id2 < dim; id2++) { // physical dimension
            for (unsigned is = 0; is < NS; is++) {
                gF2[id1 * NS + is] += gF1[id2 * NS + is]
                    * ijac(ie, iq, id1, id2) * djac(ie, iq) * quad_wts(iq);
            }
        }
    }

    // overwrite state and state gradient by fluxes
    for (unsigned is = 0; is < NS; is++) {
        Uq(ie, iq, is) = F[is];
    }
    idx = 0;
    for (unsigned id = 0; id < dim; id++) {
        for (unsigned is = 0; is < NS; is++) {
            vgUq(ie, iq, is, id) = gF2[idx];
            idx++;
        }
    }
}


template<unsigned dim>
struct InteriorFacesFluxFunctor {

public:
    InteriorFacesFluxFunctor(const Physics::Physics<dim> physics, const Mesh mesh,
        view_type_3D UqL, view_type_3D UqR, view_type_4D gUqL, view_type_4D gUqR,
        view_type_1D quad_wts, view_type_3D normals, view_type_3D Fq, Kokkos::View<int**> quad_idx_L,
        Kokkos::View<int**> quad_idx_R, unsigned rank);

  KOKKOS_INLINE_FUNCTION
    void operator()(const int iface, const int iq) const;

private:
    // set as compile time constant -> see common/defines.h
    static constexpr int NS = GLOBAL_NUM_SPECIES + 1 + dim;
    const Physics::Physics<dim> physics;
    const Mesh mesh;
    // view_type_3D Uq;
    view_type_3D UqL;
    view_type_3D UqR;
    view_type_4D gUqL;
    view_type_4D gUqR;
    view_type_3D normals;
    view_type_3D Fq;
    // view_type_2D djac;
    // view_type_4D ijac; 
    view_type_1D quad_wts;
    Kokkos::View<int**> quad_idx_L;
    Kokkos::View<int**> quad_idx_R;
    unsigned rank;
    // bool need_state_grad;

};

template<unsigned dim>
InteriorFacesFluxFunctor<dim>::InteriorFacesFluxFunctor(const Physics::Physics<dim> physics, 
    const Mesh mesh, view_type_3D UqL, view_type_3D UqR, view_type_4D gUqL, view_type_4D gUqR,
    view_type_1D quad_wts, 
    view_type_3D normals,
    view_type_3D Fq,
    Kokkos::View<int**> quad_idx_L,
    Kokkos::View<int**> quad_idx_R,
    unsigned rank) : physics{physics}, mesh{mesh}, 
            UqL{UqL}, UqR{UqR}, gUqL{gUqL}, gUqR{gUqR}, 
            quad_wts{quad_wts}, normals{normals}, 
            Fq{Fq}, quad_idx_L{quad_idx_L}, quad_idx_R{quad_idx_R}, 
            rank{rank}{
}


template<unsigned dim> KOKKOS_INLINE_FUNCTION
void InteriorFacesFluxFunctor<dim>::operator()(
    const int iface, const int iq) const {

    rtype UL[NS];
    rtype UR[NS];
    rtype F[NS];
    rtype N[dim];
    rtype gUL[dim * NS] = {0};
    rtype gUR[dim * NS] = {0};
    rtype gU_tmp[dim * NS];

    // const unsigned nqf = (unsigned)quad_wts.extent(0);
    // const unsigned rankL = mesh.get_rankL(iface);
    // const unsigned rankR = mesh.get_rankR(iface);

    // if (rank == rankL) {
    //     const unsigned elemL = mesh.get_elemL(iface);
    //     const unsigned face_ID_L = mesh.get_ref_face_idL(iface);
    //     // const unsigned elemL = mesh_local.get_local_elem_ID(elemL_global);
    //     int startL = face_ID_L * nqf;

    //     for (long unsigned is = 0; is < NS; is++){
    //         UL[is] = Uq(elemL, startL + quad_idx_L(iface, iq), is);
    //     }
    // }

    // if (rank == rankR){
    //     const unsigned elemR = mesh.get_elemR(iface);
    //     const unsigned face_ID_R = mesh.get_ref_face_idR(iface);
    //     // const unsigned elemR = mesh_local.get_local_elem_ID(elemR_global);
    //     int startR = face_ID_R * nqf;

    //     for (long unsigned is = 0; is < NS; is++){
    //         UR[is] = Uq(elemR, startR + quad_idx_R(iface, iq), is);
    //     }
    // }

    // read state from views
    for (unsigned idim = 0; idim < dim; idim++){
        N[idim] = normals(iface, iq, idim);
    }
    for (unsigned is = 0; is < NS; is++){
        UL[is] = UqL(iface, iq, is);
        UR[is] = UqR(iface, iq, is);
    }

    // TODO: Add for gradient of the state
    // read state reference-space gradients and convert to physical-space gradients
    // if (need_state_grad) {
    //     // left element
    //     for (unsigned i = 0; i < dim * NS; i++) {
    //         // YES! This is correct. See the layout constraint and how the evaluation is done.
    //         // The layout is Nf x Nq x Nd x Ns x Ne instead of Nf x Nd x Nq x Ns x Ne.
    //         gU_tmp[i] = gUqL0[str_gUL * i];
    //     }
    //     for (unsigned id1 = 0; id1 < dim; id1++) { // reference dimension
    //         for (unsigned id2 = 0; id2 < dim; id2++) { // physical dimension
    //             for (unsigned is = 0; is < NS; is++) {
    //                 const unsigned idx_x_s = id2 * NS + is;
    //                 const unsigned idx_xi_s = id1 * NS + is;
    //                 const unsigned idx_xi_x = id1 * dim + id2;
    //                 gUL[idx_x_s] += gU_tmp[idx_xi_s] * acc_ijacL[idom][idx_xi_x][p2L];
    //             }
    //         }
    //     }
    //     // right element
    //     for (unsigned i = 0; i < dim * NS; i++) {
    //         gU_tmp[i] = gUqR0[str_gUR * i];
    //     }
    //     for (unsigned id1 = 0; id1 < dim; id1++) { // reference dimension
    //         for (unsigned id2 = 0; id2 < dim; id2++) { // physical dimension
    //             for (unsigned is = 0; is < NS; is++) {
    //                 const unsigned idx_x_s = id2 * NS + is;
    //                 const unsigned idx_xi_s = id1 * NS + is;
    //                 const unsigned idx_xi_x = id1 * dim + id2;
    //                 gUR[idx_x_s] += gU_tmp[idx_xi_s] * acc_ijacR[idom][idx_xi_x][p2R];
    //             }
    //         }
    //     }
    // }


    // TODO: Get SIP stability info in this function
    // eq_input.stab = eta; // stabilization parameter for interior penalty viscous discretization
    // eq_input.hL = volL / face_area; // characteristic length of the left element
    // eq_input.hR = volR / face_area; // characteristic length of the right element

    // call the flux function
    // this will overwrite gUL and gUR by the appropriate fluxes
    physics.get_conv_flux_numerical(UL, UR, N, F, gUL, gUR);
    for (unsigned is = 0; is < NS; is++){
        // fill the iface view and multiply by quad_wts in prep for integration
        // printf("F(%i, %i, %i)=%f\n", iface, iq, is, F[is]); // before multiplying by quad_wts
        Fq(iface, iq, is) = F[is] * quad_wts(iq);
    }
}

} // end namespace FluxFunctors


#endif // DG_SOLVER_FLUX_FUNCTORS_H