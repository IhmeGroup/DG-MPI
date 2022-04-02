#ifndef DG_SOLVER_FLUX_FUNCTORS_H
#define DG_SOLVER_FLUX_FUNCTORS_H

#include "common/defines.h"
#include "physics/base/base.h"
#include <Kokkos_Core.hpp>


namespace FluxFunctors {


// I think I need to change this to something using a Kokkos Lambda
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
    // Kokkos4DView vgUq;
    // bool need_state_grad;
};

// TODO: Pass the gradient of the state to the functor
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

    // constexpr unsigned NS = Equation::NS;

    // local storage at a specific point
    rtype U[NS];
    rtype gU1[dim * NS];
    rtype gU2[dim * NS];
    rtype F[NS] = {0}; // to silence warnings
    rtype gF1[dim * NS];
    rtype gF2[dim * NS];
    unsigned idx;

    // read
    for (unsigned is = 0; is < NS; is++) {
        U[is] = Uq(ie, iq, is);
    }



    // printf("NS=%i\n", physics.get_NS());
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
                // printf("gF1(%i, %i, %i, %i)=%f\n", ie, iq, is, id2, gF1[id2 * NS + is]);
                gF2[id1 * NS + is] += gF1[id2 * NS + is]
                    * ijac(ie, iq, id1, id2) * djac(ie, iq) * quad_wts(iq);
                // printf("ijac(%i, %i, %i, %i)=%f\n", ie, iq, id1, id2, ijac(ie, iq, id1, id2));
                // printf("djac(%i, %i)=%f\n", ie, iq, djac(ie, iq));
                // printf("quad_wts(%i)=%f\n", iq, quad_wts(iq));
                // printf("gF2(%i, %i, %i, %i)=%f\n", ie, iq, is, id1, gF2[id1 * NS + is]);


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
    // printf("vgUq(0,0,0,0)=%f\n", vgUq(0,0,0,0));
}


}


#endif // DG_SOLVER_FLUX_FUNCTORS_H