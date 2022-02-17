#include <iostream>
#include <sstream>
#include <vector>
#include "io/writer.h"
#include "memory/memory_network.h"

using std::string, std::vector;


Writer::Writer(Mesh& mesh) {
    std::stringstream stream;
    stream << PROJECT_ROOT << "/build/test/mpi_enabled_tests/mesh/data.h5";
    const string file_name = stream.str();

    // Copy everything to the host from the device
    mesh.copy_from_device_to_host();

    // Write some attributes, on the head rank
    mesh.network.barrier();
    if (mesh.network.head_rank) {
        H5::H5File file(file_name, H5F_ACC_TRUNC);
        // Number of ranks
        write_attribute(mesh.network.num_ranks, "Number of Ranks", file);
        // Number of dimensions
        write_attribute(mesh.dim, "Number of Dimensions", file);
        // Number of elements
        write_attribute(mesh.num_elems, "Number of Elements", file);
        // Number of nodes
        write_attribute(mesh.num_nodes, "Number of Nodes", file);
        // Number of nodes per element
        write_attribute(mesh.num_nodes_per_elem,
                "Number of Nodes Per Element", file);
        file.close();
    }

    // Loop over each rank
    for (unsigned rank = 0; rank < mesh.network.num_ranks; rank++) {
        mesh.network.barrier();
        // Perform this in serial
        if (rank == mesh.network.rank) {
            // Open file
            H5::H5File file(file_name, H5F_ACC_RDWR);

            // Create a group for this rank
            auto group = file.createGroup("Rank " +
                    std::to_string(mesh.network.rank));

            vector<hsize_t> dimensions;
            /* -- Node coordinates -- */
            // Dimensions of the dataset: (num_nodes, dim)
            dimensions.resize(2);
            dimensions[0] = mesh.num_nodes_part;
            dimensions[1] = mesh.dim;
            write_dataset(mesh.h_node_coords.data(), "Node Coordinates", group,
                    dimensions);

            /* -- Local to global node IDs -- */
            // Dimensions of the dataset: (num_nodes)
            dimensions.resize(1);
            dimensions[0] = mesh.num_nodes_part;
            write_dataset(mesh.h_local_to_global_node_IDs.data(),
                    "Local to Global Node IDs", group, dimensions);

            /* -- Element nodes -- */
            // Dimensions of the dataset: (num_elems, num_nodes_per_elem)
            dimensions.resize(2);
            dimensions[0] = mesh.num_elems_part;
            dimensions[1] = mesh.num_nodes_per_elem;
            write_dataset(mesh.h_elem_to_node_IDs.data(),
                    "Element Global Node IDs", group, dimensions);

            /* -- Local to global element IDs -- */
            // Dimensions of the dataset: (num_elems)
            dimensions.resize(1);
            dimensions[0] = mesh.num_elems_part;
            write_dataset(mesh.h_local_to_global_elem_IDs.data(),
                    "Local to Global Element IDs", group, dimensions);

            // Close file
            file.close();
        }
    }
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
