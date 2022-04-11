#ifndef DG_SOLVER_H
#define DG_SOLVER_H

#include <string>
#include <Kokkos_Core.hpp>
#include "toml11/toml.hpp"
#include "memory/memory_network.h"
#include "numerics/numerics_data.h"
#include "mesh/mesh.h"
#include "numerics/basis/basis.h"
#include "physics/base/base.h"
#include "solver/helpers.h"
#include "numerics/timestepping/stepper.h"
#include <memory>

template<unsigned dim>
class Solver {
    public:
        Solver(const toml::value& input_file, Mesh& mesh, MemoryNetwork& network, 
            Numerics::NumericsParams& params);
        void precompute_matrix_helpers();

        void init_state_from_fcn(Mesh& mesh_local);

        void copy_from_device_to_host();

        void read_in_coefficients(const std::string& filename);

        void construct_face_states(const view_type_3D Uq, 
            view_type_3D UqL, view_type_3D UqR);

        void solve();

        void get_residual();

        void get_element_residuals();

        void get_interior_face_residuals();
        

    public:
        // Solution coefficients
        Kokkos::View<rtype***> Uc;
        host_view_type_3D h_Uc;

        // Residuals
        Kokkos::View<rtype***> res;


        // Solution evaluated at the face quadrature points. This has shape
        // (nIF, nqf, ns)
        Kokkos::View<rtype***> Uc_face;

        const toml::value& input_file;
        Mesh& mesh;
        MemoryNetwork& network;
        Numerics::NumericsParams& params;

        // Physics class
        Physics::Physics<dim> physics;
        // Basis class
        Basis::Basis basis;

        // Stepper Class
        std::shared_ptr<StepperBase<dim>> stepper;

        // Volume Helper class
        VolumeHelpers::VolumeHelperFunctor vol_helpers;
        InteriorFaceHelpers::InteriorFaceHelperFunctor iface_helpers;
        
        int nb;
        int order;

        rtype time;
};

#endif // DG_SOLVER_H
