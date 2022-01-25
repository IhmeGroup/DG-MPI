//
// Created by kihiro on 1/22/20.
//

#ifndef DG_PHYSICS_BASE_H
#define DG_PHYSICS_BASE_H

#include <string>
#include "common/defines.h"
#include "common/my_exceptions.h"

// #include "equations/equation_data.h"
// #include "physics/physics_data.h"
// #include "math/ode.h"
#include <Kokkos_Core.hpp>
#include <KokkosBlas2_gemv.hpp>

namespace Physics {

/*! \brief Base equation class
 *
 */
template <int dim> 
class PhysicsBase {
  public:
    /*! \brief Virtual destructor
     *
     * This destructor is needed so that the derived class destructors are called when releasing
     * a pointer to this base class.
     */
    virtual ~PhysicsBase() = default;
    
    virtual int get_NS();

    rtype state;

    DG_KOKKOS_FUNCTION void get_conv_flux_projected(
      Kokkos::View<const rtype*> Uq, 
      Kokkos::View<const rtype*> normals,
      Kokkos::View<rtype*> Fproj); 

    DG_KOKKOS_FUNCTION virtual void conv_flux_physical(
      Kokkos::View<const rtype*> U,
      Kokkos::View<rtype**> F);
    /*! \brief Compute an analytic state at one point / state
     *
     * @param input
     * @param type
     * @param U
     */
    // virtual void analytic_state(
    //     const EquationInput &input,
    //     const AnalyticType type,
    //     rtype *U) const = 0;

    /*! \brief Given a time-step, compute the maximum CFL at ne points
     *
     * This method assumes the following layouts:
     * - Ue: ne x ns (row-major)
     *
     * The time-step must provided in the EquationInput object (input.dt) as well as the
     * element characteristic length (input.hL).
     *
     * @param input
     * @param ne
     * @param Ue
     * @return
     */
//     virtual rtype get_maximum_CFL(
//         const EquationInput &input,
//         const unsigned ne,
//         const rtype *Ue) const = 0;

//     /*! \brief Given a varaible name, compute it at ne points
//      *
//      * This method assumes the following layouts:
//      * - Ue: ne x ns (row-major)
//      * - gUw: dim x ne x ns (row-major)
//      *
//      * @param input
//      * @param var_name
//      * @param ne
//      * @param Ue
//      * @param gUe
//      * @param v
//      */
//     virtual void get_var(
//         const EquationInput &input,
//         const std::string var_name,
//         const unsigned ne,
//         const rtype *Ue,
//         const rtype *gUe,
//         rtype *v) const = 0;

//     /*! \brief Given a variable enum, compute it at ne points
//      *
//      * This method assumes the following layouts:
//      * - Ue: ne x ns (row-major)
//      * - gUw: dim x ne x ns (row-major)
//      *
//      * @param input
//      * @param var
//      * @param ne
//      * @param Ue
//      * @param gUe
//      * @param v
//      */
//     virtual void get_var(
//         const EquationInput &input,
//         const PhysicalVariable var,
//         const unsigned ne,
//         const rtype *Ue,
//         const rtype *gUe,
//         rtype *v) const = 0;

//   public:
//     /*! \brief Compute the volume fluxes at one point
//      *
//      * @param input
//      * @param U
//      * @param gU
//      * @param F
//      * @param gF
//      */
//     virtual void get_vol_fluxes(
//         const EquationInput &input,
//         const rtype *U,
//         const rtype *gU,
//         rtype *F,
//         rtype *gF) = 0;

//     /*! \brief Compute the surface fluxes at one point
//      *
//      * @param input
//      * @param UL
//      * @param UR
//      * @param gUL
//      * @param gUR
//      * @param N
//      * @param F
//      * @param gFL
//      * @param gFR
//      */
//     virtual void get_surf_fluxes(
//         const EquationInput &input,
//         const rtype *UL,
//         const rtype *UR,
//         const rtype *gUL,
//         const rtype *gUR,
//         const rtype *N,
//         rtype *F,
//         rtype *gFL,
//         rtype *gFR) = 0;

//   public:
//     /*=====================*/
//     /* BOUNDARY CONDITIONS */
//     /*=====================*/

//     /*! \brief Get the boundary fluxes at one point for
//      *         the weak Riemann full-state boundary condition
//      *
//      * @param input
//      * @param bdata
//      * @param U
//      * @param gU
//      * @param N
//      * @param F
//      * @param gF
//      */
//     virtual void get_weakriemannfullstate_boundary_fluxes(
//         const EquationInput &input, const BoundaryData &bdata,
//         const rtype *U, const rtype *gU, const rtype *N, rtype *F, rtype *gF) {

//         throw NotImplementedException("EquationBase does not implement "
//                                       "get_weakriemannfullstate_boundary_fluxes");
//     }

//     /*! \brief Get the boundary fluxes at one point for
//      *         the weak Riemann slip-wall boundary condition
//      *
//      * @param input
//      * @param bdata
//      * @param U
//      * @param gU
//      * @param N
//      * @param F
//      * @param gF
//      */
//     virtual void get_weakriemannslipwall_boundary_fluxes(
//         const EquationInput &input, const BoundaryData &bdata,
//         const rtype *U, const rtype *gU, const rtype *N, rtype *F, rtype *gF) {

//         throw NotImplementedException("EquationBase does not implement "
//                                       "get_weakriemannslipwall_boundary_fluxes");
//     }

//     /*! \brief Get the boundary fluxes at one point for
//      *         the weak Riemann pressure outlet boundary condition
//      *
//      * @param input
//      * @param bdata
//      * @param U
//      * @param gU
//      * @param N
//      * @param F
//      * @param gF
//      */
//     virtual void get_weakriemannpressureoutlet_boundary_fluxes(
//         const EquationInput &input, const BoundaryData &bdata,
//         const rtype *U, const rtype *gU, const rtype *N, rtype *F, rtype *gF) {

//         throw NotImplementedException("EquationBase does not implement "
//                                       "get_weakriemannpressureoutlet_boundary_fluxes");
//     }

//     /*! \brief Get the boundary fluxes at one point for
//      *         the weak prescribed pressure outlet boundary condition
//      *
//      * @param input
//      * @param bdata
//      * @param U
//      * @param gU
//      * @param N
//      * @param F
//      * @param gF
//      */
//     virtual void get_weakprescribedpressureoutlet_boundary_fluxes(
//         const EquationInput &input, const BoundaryData &bdata,
//         const rtype *U, const rtype *gU, const rtype *N, rtype *F, rtype *gF) {

//         throw NotImplementedException("EquationBase does not implement "
//                                       "get_weakprescribedpressureoutlet_boundary_fluxes");
//     }

//     virtual void get_weakprescribedisothermalwall_boundary_fluxes(
//          const EquationInput &input, const BoundaryData &bdata,
//          const rtype *U, const rtype *gU, const rtype *N, rtype *F, rtype *gF) {

//      throw NotImplementedException("EquationBase does not implement "
//                                    "get_weakprescribedisothermalwall_boundary_fluxes");
//     }
//     virtual void get_weakriemannisothermalwall_boundary_fluxes(
//          const EquationInput &input, const BoundaryData &bdata,
//          const rtype *U, const rtype *gU, const rtype *N, rtype *F, rtype *gF) {

//      throw NotImplementedException("EquationBase does not implement "
//                                    "get_weakriemannisothermalwall_boundary_fluxes");
//     }
//     virtual void get_weakprescribedisothermalvelwall_boundary_fluxes(
//          const EquationInput &input, const BoundaryData &bdata,
//          const rtype *U, const rtype *gU, const rtype *N, rtype *F, rtype *gF) {

//      throw NotImplementedException("EquationBase does not implement "
//                                    "get_weakprescribedisothermalvelwall_boundary_fluxes");
//     }

//      virtual void get_hardfullstate_boundary_fluxes(
//          const EquationInput &input,
//          const BoundaryData &bdata,
//          const rtype *U,
//          const rtype *gU,
//          const rtype *N,
//          rtype *F,
//          rtype *gF) {

//          throw NotImplementedException("EquationBase does not implement "
//                                        "get_hardfullstate_boundary_fluxes");
//     }

//     virtual void get_stegerwarmingfarfield_boundary_fluxes(
//         const EquationInput &input,
//         const BoundaryData &bdata,
//         const rtype *U,
//         const rtype *gU,
//         const rtype *N,
//         rtype *F,
//         rtype *gF) {

//         throw NotImplementedException("EquationBase does not implement "
//                                       "get_stegerwarmingfarfield_boundary_fluxes");
//     }

//     virtual void get_weakriemanncustom_boundary_fluxes(
//         const EquationInput &input,
//         const BoundaryData &bdata,
//         const rtype *U,
//         const rtype *gU,
//         const rtype *N,
//         rtype *F,
//         rtype *gF) {

//         throw NotImplementedException("EquationBase does not implement "
//                                       "get_weakriemanncustom_boundary_fluxes");
//     }

//     virtual void get_symmetry_boundary_fluxes(
//         const EquationInput &input,
//         const BoundaryData &bdata,
//         const rtype *U,
//         const rtype *gU,
//         const rtype *N,
//         rtype *F,
//         rtype *gF) {

//         throw NotImplementedException("EquationBase does not implement "
//                                       "get_symmetry_boundary_fluxes");
//     }

//   public:
//     /*============================================*/
//     /* INTERFACE FOR CHEMICAL REACTION STIFF STEP */
//     /*============================================*/

//     /*! \brief Integrate the chemical ODE system at one point by dt in time
//      *
//      * @param data
//      * @param dt
//      * @param Uq
//      */
//     virtual void reaction_step(const Math::ROWPlusParams &data, const rtype dt, rtype *Uq) {
//         throw NotImplementedException("EquationBase does not implement reaction_step.");
//     }

//   protected:
//     unsigned ns; //!< runtime number of state variables
// };

// /*=======================================*/
// /* BOUNDARY CONDITIONS WRAPPER FUNCTIONS */
// /*=======================================*/

// /* I chose this method for now because C++ does not allow virtual member template. Indeed, the
//  * number of entries in the virtual function table needs to be fixed.
//  * I put this structure and wrapper functions here since the boundary conditions are part of
//  * the equation at the end of the day. */

// template<BoundaryType type>
// void get_boundary_fluxes(EquationBase *equation,
//     const EquationInput &input, const Equation::BoundaryData &bdata,
//     const rtype *U, const rtype *gU, const rtype *N, rtype *F, rtype *gF);

// template<> inline
// void get_boundary_fluxes<BoundaryType::WeakRiemannFullState>(EquationBase *equation,
//     const EquationInput &input, const Equation::BoundaryData &bdata,
//     const rtype *U, const rtype *gU, const rtype *N, rtype *F, rtype *gF) {

//     equation->get_weakriemannfullstate_boundary_fluxes(input, bdata, U, gU, N, F, gF);
// }

// template<> inline
// void get_boundary_fluxes<BoundaryType::WeakRiemannSlipWall>(EquationBase *equation,
//     const EquationInput &input, const Equation::BoundaryData &bdata,
//     const rtype *U, const rtype *gU, const rtype *N, rtype *F, rtype *gF) {

//     equation->get_weakriemannslipwall_boundary_fluxes(input, bdata, U, gU, N, F, gF);
// }

// template<> inline
// void get_boundary_fluxes<BoundaryType::WeakPrescribedPressureOutlet>(EquationBase *equation,
//     const EquationInput &input, const Equation::BoundaryData &bdata,
//     const rtype *U, const rtype *gU, const rtype *N, rtype *F, rtype *gF) {

//     equation->get_weakprescribedpressureoutlet_boundary_fluxes(input, bdata, U, gU, N, F, gF);
// }

// template<> inline
// void get_boundary_fluxes<BoundaryType::WeakRiemannPressureOutlet>(EquationBase *equation,
//     const EquationInput &input, const Equation::BoundaryData &bdata,
//     const rtype *U, const rtype *gU, const rtype *N, rtype *F, rtype *gF) {

//     equation->get_weakriemannpressureoutlet_boundary_fluxes(input, bdata, U, gU, N, F, gF);
// }

// template<> inline
// void get_boundary_fluxes<BoundaryType::HardFullState>(
//     EquationBase *equation,
//     const EquationInput &input,
//     const BoundaryData &bdata,
//     const rtype *U,
//     const rtype *gU,
//     const rtype *N,
//     rtype *F,
//     rtype *gF) {

//     equation->get_hardfullstate_boundary_fluxes(input, bdata, U, gU, N, F, gF);
// }

// template<> inline
// void get_boundary_fluxes<BoundaryType::StegerWarmingFarField>(
//     EquationBase *equation,
//     const EquationInput &input,
//     const BoundaryData &bdata,
//     const rtype *U,
//     const rtype *gU,
//     const rtype *N,
//     rtype *F,
//     rtype *gF) {

//     equation->get_stegerwarmingfarfield_boundary_fluxes(input, bdata, U, gU, N, F, gF);
// }

// template<> inline
// void get_boundary_fluxes<BoundaryType::WeakRiemannCustom>(
//     EquationBase *equation,
//     const EquationInput &input,
//     const BoundaryData &bdata,
//     const rtype *U,
//     const rtype *gU,
//     const rtype *N,
//     rtype *F,
//     rtype *gF) {

//     equation->get_weakriemanncustom_boundary_fluxes(input, bdata, U, gU, N, F, gF);
// }

// template<> inline
// void get_boundary_fluxes<BoundaryType::Symmetry>(
//     EquationBase *equation,
//     const EquationInput &input,
//     const BoundaryData &bdata,
//     const rtype *U,
//     const rtype *gU,
//     const rtype *N,
//     rtype *F,
//     rtype *gF) {

//     equation->get_symmetry_boundary_fluxes(input, bdata, U, gU, N, F, gF);
// }

// template<> inline
// void get_boundary_fluxes<BoundaryType::WeakPrescribedIsothermalWall>(EquationBase *equation,
//     const EquationInput &input, const Equation::BoundaryData &bdata,
//     const rtype *U, const rtype *gU, const rtype *N, rtype *F, rtype *gF) {

//     equation->get_weakprescribedisothermalwall_boundary_fluxes(input, bdata, U, gU, N, F, gF);
// }
// template<> inline
// void get_boundary_fluxes<BoundaryType::WeakRiemannIsothermalWall>(EquationBase *equation,
//     const EquationInput &input, const Equation::BoundaryData &bdata,
//     const rtype *U, const rtype *gU, const rtype *N, rtype *F, rtype *gF) {

//  equation->get_weakriemannisothermalwall_boundary_fluxes(input, bdata, U, gU, N, F, gF);
// }
// template<> inline
// void get_boundary_fluxes<BoundaryType::WeakPrescribedTempVelWall>(EquationBase *equation,
//     const EquationInput &input, const Equation::BoundaryData &bdata,
//     const rtype *U, const rtype *gU, const rtype *N, rtype *F, rtype *gF) {

//     equation->get_weakprescribedisothermalvelwall_boundary_fluxes(input, bdata, U, gU, N, F, gF);
};

} // namespace Physics

#endif //DG_PHYSICS_BASE_H