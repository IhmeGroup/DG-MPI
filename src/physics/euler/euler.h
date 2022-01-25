#ifndef DG_EULER_H
#define DG_EULER_H

#include <algorithm>
#include <memory>
#include "common/defines.h"
// #include "equations/analytic_expressions.h"
// #include "math/linear_algebra_hand_written.h"
#include "physics/base/base.h"
// #include "equations/num_fluxes.h"
// #include "physics/ideal_gas.h"

#include <Kokkos_Core.hpp>
#include <KokkosBlas1_dot.hpp>

namespace Physics {

/*
  This class corresponds to the compressible Euler equations for a
  calorically perfect gas. It inherits attributes and methods from the
  PhysicsBase class. See PhysicsBase for detailed comments of attributes
  and methods.

  Additional methods and attributes are commented below.
*/
template <int dim>
class Euler : public PhysicsBase<dim> {
  public:
    int get_NS();
    rtype gamma;
    rtype R;
  public:
    /*
    Set Euler physical parameters

    Inputs:
    -------
      GasConstant: mass-specific gas constant
      SpecificHeatRatio: ratio of specific heats
    */
    void set_physical_params(rtype GasConstant=287.0, rtype SpecificHeatRatio=1.4);

    /*
    Get pressure

    Inputs:
    -------
      U: solution state [ns]
    
    Outputs:
    --------
      pressure
    */
    DG_KOKKOS_FUNCTION rtype get_pressure(Kokkos::View<const rtype*> U);


    DG_KOKKOS_FUNCTION void conv_flux_physical(
        Kokkos::View<const rtype*> U,
        Kokkos::View<rtype**> F);


    // static void conv_flux_normal(const rtype *U, const rtype P, const rtype *N, rtype *F);
};

  // public:
  //   DG_KOKKOS_INLINE_FUNCTION static void vol_fluxes(
  //       const EquationInput &input,
  //       const Physics::IdealGasParams &params,
  //       const rtype *U,
  //       const rtype *gU,
  //       rtype *F,
  //       rtype *gF);
  // protected:
    // DG_KOKKOS_INLINE_FUNCTION static void Fc(
  //       const Physics::IdealGasParams &params,
  //       const rtype *U,
  //       rtype *gF);

  // public:
    // explicit EulerBase();
//   public:
//     virtual void analytic_state(
//         const EquationInput &input,
//         const AnalyticType type,
//         rtype *U) const override;
//     virtual rtype get_maximum_CFL(
//         const EquationInput &input,
//         const unsigned ne,
//         const rtype *Ue) const override;
//     virtual void get_var(
//         const EquationInput &input,
//         const std::string var_name,
//         const unsigned ne,
//         const rtype *Ue,
//         const rtype *gUe,
//         rtype *v) const override;
//     virtual void get_var(
//         const EquationInput &input,
//         const PhysicalVariable var,
//         const unsigned ne,
//         const rtype *Ue,
//         const rtype *gUe,
//         rtype *v) const override;
//   public:
//     virtual void get_vol_fluxes(
//         const EquationInput &input,
//         const rtype *U,
//         const rtype *gU,
//         rtype *F,
//         rtype *gF) override;
//   public:
//     virtual void get_weakprescribedpressureoutlet_boundary_fluxes(
//         const EquationInput &input,
//         const BoundaryData &bdata,
//         const rtype *U,
//         const rtype *gU,
//         const rtype *N,
//         rtype *F,
//         rtype *gF) override;

//   protected:
//     /*! \brief Get the convective flux
//      *
//      * @param stride_state
//      * @param stride_dim
//      * @param U
//      * @param Fc
//      */
//     virtual void get_Fc(const rtype *U, rtype *Fc);

//     /*! \brief Get the convective flux dotted with N
//      *
//      * @param stride_state
//      * @param U
//      * @param N
//      * @param Fcn
//      */
//     virtual void get_Fcn(
//         const unsigned stride_state,
//         const rtype *U,
//         const rtype *N,
//         rtype *Fcn);

//   protected:
//     mutable Physics::IdealGas<dim> phys;
// };

// template <unsigned dim, InviscNumFlux flux>
// class Euler : public EulerBase<dim> {
  // public:
    // static constexpr unsigned NS = dim + 2;

//   public:
//     static std::string get_name();
//     DG_KOKKOS_INLINE_FUNCTION static void vol_fluxes(
//         const EquationInput &input,
//         const Physics::IdealGasParams &params,
//         const rtype *U,
//         const rtype *gU,
//         rtype *F,
//         rtype *gF);
//     DG_KOKKOS_INLINE_FUNCTION static void surf_fluxes(
//         const EquationInput &input,
//         const Physics::IdealGasParams &params,
//         const rtype *UL,
//         const rtype *UR,
//         const rtype *N,
//         rtype *F,
//         rtype *gUL,
//         rtype *gUR);

//   protected:
//     DG_KOKKOS_INLINE_FUNCTION static void Fchat(
//         const Physics::IdealGasParams &params,
//         const rtype *UL,
//         const rtype *UR,
//         const rtype *N,
//         rtype *F);

//   public:
//     explicit Euler(const Physics::IdealGasParams &data);
//   public:
//     virtual void get_surf_fluxes(
//         const EquationInput &input,
//         const rtype *UL,
//         const rtype *UR,
//         const rtype *gUL,
//         const rtype *gUR,
//         const rtype *N,
//         rtype *F,
//         rtype *gFL,
//         rtype *gFR) override;
//   public:
//     virtual void get_weakriemannfullstate_boundary_fluxes(
//         const EquationInput &input,
//         const BoundaryData &bdata,
//         const rtype *U,
//         const rtype *gU,
//         const rtype *N,
//         rtype *F,
//         rtype *gF) override;
//     virtual void get_weakriemannslipwall_boundary_fluxes(
//         const EquationInput &input,
//         const BoundaryData &bdata,
//         const rtype *U,
//         const rtype *gU,
//         const rtype *N,
//         rtype *F,
//         rtype *gF) override;
//     virtual void get_weakriemannpressureoutlet_boundary_fluxes(
//         const EquationInput &input,
//         const BoundaryData &bdata,
//         const rtype *U,
//         const rtype *gU,
//         const rtype *N,
//         rtype *F,
//         rtype *gF) override;
//   protected:
//     /* Wrapper around numerical flux kernels (pure functions) to allow for different interfaces
//      * for the kernels (for example, split form fluxes). The role of this method is to compute
//      * all inputs for the kernels. This method, common for all concrete classes, can then be reused
//      * transparently in other methods such as those for boundary conditions. */
//     virtual void get_Fchat(
//         const rtype *UL,
//         const rtype *UR,
//         const rtype *N,
//         rtype *Fchat);
// };

} // namespace Physics

// NOTE: Keep just in case we go back to *_impl.h version
// #include "physics/euler/euler.cpp"

#endif //DG_EULER_H