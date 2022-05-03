#ifndef DG_WRITER_H
#define DG_WRITER_H

#include <string>
#include <vector>
#include "mesh/mesh.h"
#include "memory/memory_network.h"
#include "solver/base.h"
#include "reader.h"


class Writer {
    public:
        Writer(const std::string& name, Mesh& mesh, MemoryNetwork& network, host_view_type_3D h_Uc,
            int nb, int ns, rtype time, bool parallel=true);

        Writer(const std::string& name, const Reader& reader, bool parallel=true);

        Writer();

        void write(
            const std::string& name,
            unsigned num_ranks,
            unsigned dim,
            unsigned num_elems,
            unsigned num_nodes,
            unsigned num_nodes_per_elem,
            unsigned nb,
            unsigned ns,
            double time,
            unsigned num_elems_part,
            bool stored_layout ,
            const double* node_coords,
            const std::vector<hsize_t>& node_coords_dim,
            const unsigned* local_to_global_node_IDs,
            const std::vector<hsize_t>& local_to_global_node_IDs_dim,
            const unsigned* elem_to_node_IDs,
            const std::vector<hsize_t>& elem_to_node_IDs,
            const unsigned* local_to_global_elem_IDs,
            const std::vector<hsize_t>& local_to_global_elem_IDs_dim,
            const double* Uc,
            const std::vector<hsize_t>& Uc_dim,
            bool parallel=true
        );
};

#endif //DG_WRITER_H
