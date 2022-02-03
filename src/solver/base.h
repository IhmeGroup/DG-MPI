#ifndef DG_SOLVER_H
#define DG_SOLVER_H

#include <Kokkos_Core.hpp>
#include "toml11/toml.hpp"
#include "memory/memory_network.h"
#include "mesh/mesh.h"

class Solver {
    public:
        Solver(const toml::value& input_file, Mesh& mesh, MemoryNetwork& network);
    public:
        // Solution coefficients
        Kokkos::View<rtype***> Uc;
        // Solution evaluated at the face quadrature points. This has shape
        // (nIF, nqf, ns)
        Kokkos::View<rtype***> U_face;
        const toml::value& input_file;
        Mesh& mesh;
        MemoryNetwork& network;
};

#endif // DG_SOLVER_H
