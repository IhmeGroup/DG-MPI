#include <iostream>
#include <vector>
#include "io/writer.h"

using std::string, std::vector;


Writer::Writer(const Mesh& mesh) {
    // /* We write one restart file overall. Only point task 0 creates a file. The other point tasks
    //  * return immediately. The top-level task must wait that point task 0 returned before
    //  * attempting to write to the restart file. */
    // //const string file_name = Utils::get_fname_with_iter(
    // //    inputs.restart_file_prefix, ".h5", time_info->iter_curr);
    // // TODO: get file name
    // const string file_name = string(PROJECT_ROOT) + "/build/test/mpi_enabled_tests/mesh/data.h5";

    // // Write some attributes, on the head rank
    // mesh.network.barrier();
    // if (mesh.network.head_rank) {
    //     H5::H5File file(file_name, H5F_ACC_TRUNC);
    //     // Number of ranks
    //     write_attribute(mesh.network.num_ranks, "Number of Ranks", file);
    //     // Number of dimensions
    //     write_attribute(mesh.dim, "Number of Dimensions", file);
    //     // Number of elements
    //     write_attribute(mesh.num_elems, "Number of Elements", file);
    //     // Number of nodes
    //     write_attribute(mesh.num_nodes, "Number of Nodes", file);
    //     // Number of nodes per element
    //     write_attribute(mesh.num_nodes_per_elem,
    //             "Number of Nodes Per Element", file);
    //     file.close();
    // }

    // // Loop over each rank
    // for (int rank = 0; rank < mesh.network.num_ranks; rank++) {
    //     mesh.network.barrier();
    //     // Perform this in serial
    //     if (rank == mesh.network.rank) {
    //         // Open file
    //         H5::H5File file(file_name, H5F_ACC_RDWR);

    //         // Create a group for this rank
    //         auto group = file.createGroup("Rank " +
    //                 std::to_string(mesh.network.rank));

    //         vector<hsize_t> dimensions;
    //         /* -- Node coordinates -- */
    //         // Dimensions of the dataset: (num_nodes, dim)
    //         dimensions = vector<hsize_t>({mesh.num_nodes_part, mesh.dim});
    //         write_dataset(mesh.node_coords.data(), "Node Coordinates", group,
    //                 dimensions);

    //         /* -- Local to global node IDs -- */
    //         // Dimensions of the dataset: (num_nodes)
    //         dimensions = vector<hsize_t>({mesh.num_nodes_part});
    //         write_dataset(mesh.local_to_global_node_IDs.data(),
    //                 "Local to Global Node IDs", group, dimensions);

    //         /* -- Element nodes -- */
    //         // Dimensions of the dataset: (num_elems, num_nodes_per_elem)
    //         dimensions = vector<hsize_t>({mesh.num_elems_part,
    //                 mesh.num_nodes_per_elem});
    //         write_dataset(mesh.elem_to_node_IDs.data(),
    //                 "Element Global Node IDs", group, dimensions);

    //         /* -- Local to global element IDs -- */
    //         // Dimensions of the dataset: (num_elems)
    //         dimensions = vector<hsize_t>({mesh.num_elems_part});
    //         write_dataset(mesh.local_to_global_elem_IDs.data(),
    //                 "Local to Global Element IDs", group, dimensions);

    //         // Close file
    //         file.close();
    //     }
    // }
}

template <class T>
void Writer::write_dataset(T data, string name, H5::Group group,
        vector<hsize_t> dimensions) {
    // Get HDF5 type
    auto type = get_hdf5_type<T>();
    // Create dataspace
    H5::DataSpace dataspace(dimensions.size(), dimensions.data());
    // Create dataset
    auto dataset = group.createDataSet(name, type, dataspace);
    // Write
    dataset.write(data, type);
}

template <class T>
void Writer::write_attribute(T data, string name, H5::H5File file) {
    // Get HDF5 type
    auto type = get_hdf5_type<T>();
    // Create dataspace
    H5::DataSpace dataspace(H5S_SCALAR);
    // Create attribute
    auto attribute = file.createAttribute(name, type, dataspace);
    // Write
    attribute.write(type, (void*)&data);
}

template <>
H5::PredType Writer::get_hdf5_type<double*>() {
    return H5::PredType::NATIVE_DOUBLE;
}

template <>
H5::PredType Writer::get_hdf5_type<int*>() {
    return H5::PredType::NATIVE_INT;
}

template <>
H5::PredType Writer::get_hdf5_type<int>() {
    return H5::PredType::NATIVE_INT;
}

template <>
H5::PredType Writer::get_hdf5_type<unsigned>() {
    return H5::PredType::NATIVE_UINT;
}
