#ifndef DG_SOLVER_H
#define DG_SOLVER_H

#include <Kokkos_Core.hpp>
#include "toml11/toml.hpp"
#include "memory/memory_network.h"
#include "numerics/numerics_data.h"
#include "mesh/mesh.h"
#include "numerics/basis/basis.h"
#include "solver/helpers.h"

class Solver {
    public:
        Solver(const toml::value& input_file, Mesh& mesh, MemoryNetwork& network,
            Numerics::NumericsParams& params);
        void precompute_matrix_helpers();
    public:
        // Solution coefficients
        Kokkos::View<rtype***> Uc;
        // Solution evaluated at the face quadrature points. This has shape
        // (nIF, nqf, ns)
        Kokkos::View<rtype***> U_face;
        const toml::value& input_file;
        Mesh& mesh;
        MemoryNetwork& network;
        Numerics::NumericsParams& params;

        // Basis class
        Basis::Basis basis;

        // Volume Helper class
        VolumeHelpers::VolumeHelperFunctor vol_helpers;
        
        int nb;
        int order;
};

#endif // DG_SOLVER_H
