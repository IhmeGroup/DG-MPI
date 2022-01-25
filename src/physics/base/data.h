//
// Created by kihiro on 7/24/20.
//

#ifndef DG_PHYSICS_DATA_H
#define DG_PHYSICS_DATA_H

#include "common/defines.h"
#include "common/my_exceptions.h"
#include "physics/base/base.h"

class FcnBase {
    public:
        virtual ~FcnBase() = default;

        /*
        Get state function for the base class

        This operatures as a virtual function for get_state. If get_state is not defined
        in each class that inherits FcnBase, it prints an error.
        */
        template<unsigned dim>
        void get_state(Physics::PhysicsBase<dim> &physics, const rtype *x, const rtype *t);
};

// enum class AnalyticType {
//     Uniform,
//     None,
// };

// inline AnalyticType get_analytic_type(const std::string &name) {
//     if (name == "Uniform") {
//         return AnalyticType::Uniform;
//     }
//     return AnalyticType::None;
// }

// enum class SourceType {
//     ManufacturedNS,
//     Channel,
//     None,
// };

// inline SourceType get_source_type(const std::string &name) {
//     if (name == "ManufacturedNS") {
//         return SourceType::ManufacturedNS;
//     }
//     if (name == "Channel") {
//         return SourceType::Channel;
//     }
//     return SourceType::None;
// // }

// enum class PhysicsType {
//     Euler,
//     None,
// };

// inline PhysicsType get_physics_type(
//     const std::string &name)
//     // const std::string &inv_flux_name,
//     // const std::string &visc_flux_name) 
// {

//     if (name == "Euler") {
//         // if (inv_flux_name == "HLLC") {
//         return PhysicsType::Euler;
//         // }
//         // if (inv_flux_name == "Roe") {
//             // return EquationType::EulerRoe;
//         // }
//     }
//     // if (name == "NavierStokes") {
//     //     if (inv_flux_name == "HLLC") {
//     //         if (visc_flux_name == "IP") {
//     //             return EquationType::NavierStokesHLLC;
//     //         }
//     //     }
//     //     if (inv_flux_name == "Roe") {
//     //         return EquationType::NavierStokesRoe;
//     //     }
//     // }
//     // if (name == "EulerMultiSpecies") {
//     //     return EquationType::EulerMultiSpecies;
//     // }
//     // if (name == "NavierStokesMultiSpecies") {
//     //     return EquationType::NavierStokesMultiSpecies;
//     // }
//     return PhysicsType::None;
// }

// enum class BoundaryType {
//     WeakRiemannSlipWall,
//     WeakRiemannIsothermalWall,
//     WeakPrescribedIsothermalWall,
//     WeakPrescribedTempVelWall,
//     WeakRiemannFullState,
//     WeakPrescribedPressureOutlet,
//     WeakRiemannPressureOutlet,
//     HardFullState,
//     StegerWarmingFarField,
//     WeakRiemannCustom,
//     Symmetry,
//     None,
// };

// inline BoundaryType get_boundary_type(const std::string &name) {
//     if (name == "WeakRiemannSlipWall") {
//         return BoundaryType::WeakRiemannSlipWall;
//     }
//     if (name == "WeakRiemannIsothermalWall") {
//         return BoundaryType::WeakRiemannIsothermalWall;
//     }
//     if (name == "WeakPrescribedIsothermalWall") {
//         return BoundaryType::WeakPrescribedIsothermalWall;
//     }
//     if (name == "WeakPrescribedTempVelWall") {
//         return BoundaryType::WeakPrescribedTempVelWall;
//     }
//     if (name == "WeakRiemannFullState") {
//         return BoundaryType::WeakRiemannFullState;
//     }
//     if (name == "WeakPrescribedPressureOutlet") {
//         return BoundaryType::WeakPrescribedPressureOutlet;
//     }
//     if (name == "WeakRiemannPressureOutlet") {
//         return BoundaryType::WeakRiemannPressureOutlet;
//     }
//     if (name == "HardFullState") {
//         return BoundaryType::HardFullState;
//     }
//     if (name == "StegerWarmingFarField") {
//         return BoundaryType::StegerWarmingFarField;
//     }
//     if (name == "WeakRiemannCustom") {
//         return BoundaryType::WeakRiemannCustom;
//     }
//     if (name == "Symmetry") {
//         return BoundaryType::Symmetry;
//     }
//     return BoundaryType::None;
// }

// enum class InviscNumFlux {
//     HLLC,
//     Roe,
// };

// enum class ViscNumFlux {
//     IP,
//     BR2,
// };

// namespace Equation {

// /*! \brief Boundary face group data from input file
//  *
//  */
// struct BoundaryData {
//     BoundaryType type;
//     AnalyticType analytic_type = AnalyticType::None;
//     rtype data[N_BDATA_MAX];
//     unsigned nData = 0;
//     bool has_data = false;
// };

// /*! \brief Equation data from input file
//  *
//  */
// struct EquationParams {
//   public:
//     bool need_state_grad() const {
//         return
//             type == EquationType::NavierStokesHLLC ||
//             type == EquationType::NavierStokesRoe ||
//             type == EquationType::NavierStokesMultiSpecies;
//     }

//     bool has_vol_term_with_basis_value() const {
//         return src_t != SourceType::None;
//     }

//     bool has_vol_term_with_basis_grad() const {
//         return true;
//     }

//     bool has_face_term_with_basis_value() const {
//         return true;
//     }

//     bool has_face_term_with_basis_grad() const {
//         return
//             type == EquationType::NavierStokesHLLC ||
//             type == EquationType::NavierStokesRoe ||
//             type == EquationType::NavierStokesMultiSpecies;
//     }

//   public:
//     EquationType type;
//     AnalyticType init_t;
//     SourceType src_t;
//     rtype init_ex_params[INIT_EX_PARAMS_MAX];
//     int nBFG;
//     BoundaryData bdata[N_BFG_MAX];
// };

// /*! \brief Extensible structure to provide to Equation objects
//  *
//  * The purpose of this structure is to provide arguments to Equation objects instead of
//  * adding additional arguments to their interface. It should be only provided as a const
//  * reference.
//  */
// struct EquationInput {
//     unsigned stride_state; //!< stride between data for different state variables
//     unsigned stride_dim; //!< stride between data for different spatial dimension componenent

//     rtype hL; //!< size of the (left) element
//     rtype hR; //!< size of the right element
//     rtype stab; //!< stabilization parameter for the visous discretization

//     SourceType source_t = SourceType::None; //!< source term type if any (for volume fluxes)
//     bool include_reaction = false; //!< whether to include the reaction step without splitting

//     rtype t; //!< current time
//     rtype dt; //!< current time-step

//     const rtype *X = nullptr; //!< pointer to constant position data
//     const rtype *ex_params = nullptr; //!< ?? TODO Steven please fill this
// };

// } // namespace Equation

#endif //DG_PHYSICS_DATA_H