#include <iostream>
#include <sstream>
#include <vector>
#include "io/writer.h"
#include "memory/memory_network.h"
#include "HDF5Wrapper.h"

Writer::Writer() {};

void Writer::write(
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
        const std::vector<hsize_t>& elem_to_node_IDs_dim,
        const unsigned* local_to_global_elem_IDs,
        const std::vector<hsize_t>& local_to_global_elem_IDs_dim,
        const double* Uc,
        const std::vector<hsize_t>& Uc_dim,
        bool parallel
        )
{

    HDF5File file;
    int mpi_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    if (parallel || mpi_rank == HDF5File::head_rank)
    {
        file.open(name, H5F_ACC_TRUNC, parallel);

        // Write some attributes to the file
        file.create_and_write_attribute("Number of Ranks", num_ranks);
        file.create_and_write_attribute("Number of Dimensions", dim);
        file.create_and_write_attribute("Number of Elements", num_elems);
        file.create_and_write_attribute("Number of Nodes", num_nodes);
        file.create_and_write_attribute("Number of Nodes Per Element", num_nodes_per_elem);
        file.create_and_write_attribute("Number of Basis Functions", nb);
        file.create_and_write_attribute("Number of State Variables", ns);
        file.create_and_write_attribute("Solver Final Time", time);

        file.create_and_write_dataset_gather_scalar("Number of Elements per Partition", num_elems_part);

        file.create_and_write_attribute("Stored Layout", stored_layout);

        file.create_and_write_parallel_dataset("Node Coordinates", node_coords_dim, node_coords);
        file.create_and_write_parallel_dataset("Local to Global Node ID", local_to_global_node_IDs_dim, local_to_global_node_IDs);
        file.create_and_write_parallel_dataset("Element Global Node IDs", elem_to_node_IDs_dim, elem_to_node_IDs);
        file.create_and_write_parallel_dataset("Local to Global Element IDs", local_to_global_elem_IDs_dim, local_to_global_elem_IDs);
        file.create_and_write_parallel_dataset("Solution Coefficients", Uc_dim, Uc);
    }
}



Writer::Writer(const std::string& name, Mesh& mesh, MemoryNetwork& network, host_view_type_3D h_Uc, int nb, int ns, rtype time, bool parallel)
{
    mesh.copy_from_device_to_host();

    bool stored_layout = std::is_same_v<decltype(h_Uc)::array_layout,Kokkos::LayoutRight>;

    write(
            name,
            network.num_ranks,
            mesh.dim,
            mesh.num_elems,
            mesh.num_nodes,
            mesh.num_nodes_per_elem,
            nb,
            ns,
            time,
            mesh.num_elems_part,
            stored_layout,
            mesh.h_node_coords.data(),
            {static_cast<hsize_t>(mesh.num_nodes_part), static_cast<hsize_t>(mesh.dim)},
            mesh.h_local_to_global_node_IDs.data(),
            {static_cast<hsize_t>(mesh.num_nodes_part)},
            mesh.h_elem_to_node_IDs.data(),
            {static_cast<hsize_t>(mesh.num_elems_part), static_cast<hsize_t>(mesh.num_nodes_per_elem)},
            mesh.h_local_to_global_elem_IDs.data(),
            {static_cast<hsize_t>(mesh.num_elems_part)},
            h_Uc.data(),
            {static_cast<hsize_t>(mesh.num_elems_part), static_cast<hsize_t>(nb), static_cast<hsize_t>(ns)},
            parallel
        );
}


Writer::Writer(const std::string& name, const Reader& reader, bool parallel)
{
    if (reader.serialize && parallel)
    {
        std::cout<<"ERROR: cannot write a serialized dataset in parallel"<<std::endl;
        std::exit(EXIT_FAILURE);
    }

    write(
        name,
        reader.num_ranks,
        reader.dim,
        reader.num_elems,
        reader.num_nodes,
        reader.num_nodes_per_elem,
        reader.nb,
        reader.ns,
        reader.time,
        reader.num_elems_part,
        reader.stored_layout,
        reader.node_coords.data.data(), reader.node_coords.dimensions,
        reader.local_to_global_node_IDs.data.data(), reader.local_to_global_node_IDs.dimensions,
        reader.elem_to_node_IDs.data.data(), reader.elem_to_node_IDs.dimensions,
        reader.local_to_global_elem_IDs.data.data(), reader.local_to_global_elem_IDs.dimensions,
        reader.Uc.data.data(), reader.Uc.dimensions,
        parallel);

}

