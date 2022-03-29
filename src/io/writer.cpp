#include <iostream>
#include <sstream>
#include <vector>
#include "io/writer.h"
#include "memory/memory_network.h"

using std::string, std::vector;

Writer::Writer(Mesh& mesh, MemoryNetwork& network, host_view_type_3D h_Uc,
    int nb, int ns) {
    std::stringstream stream;
    // stream << PROJECT_ROOT << "/build_gpu/test/mpi_enabled_tests/mesh/data.h5";
    stream << "data.h5";
    const string file_name = stream.str();

    // Copy mesh info to the host from the device
    mesh.copy_from_device_to_host();

    // Write some attributes, on the head rank
    network.barrier();
    if (network.head_rank) {
        H5::H5File file(file_name, H5F_ACC_TRUNC);
        // Number of ranks
        write_attribute(network.num_ranks, "Number of Ranks", file);
        // Number of dimensions
        write_attribute(mesh.dim, "Number of Dimensions", file);
        // Number of elements
        write_attribute(mesh.num_elems, "Number of Elements", file);
        // Number of nodes
        write_attribute(mesh.num_nodes, "Number of Nodes", file);
        // Number of nodes per element
        write_attribute(mesh.num_nodes_per_elem,
                "Number of Nodes Per Element", file);
        write_attribute(nb,
                "Number of Basis Functions", file);
        write_attribute(ns,
                "Number of State Variables", file);        
        file.close();
    }

    // Loop over each rank
    for (unsigned rank = 0; rank < network.num_ranks; rank++) {
        network.barrier();
        // Perform this in serial
        if (rank == network.rank) {
            // Open file
            H5::H5File file(file_name, H5F_ACC_RDWR);

            // Create a group for this rank
            auto group = file.createGroup("Rank " +
                    std::to_string(network.rank));


            // Number of elements per partition
            write_attribute(mesh.num_elems_part, 
                "Number of Elements per Partition", group);
            // Get the layout to store
            bool stored_layout = 
                std::is_same_v<decltype(h_Uc)::array_layout,Kokkos::LayoutRight>;
            // store the layout of Uc
            write_attribute(stored_layout, "Stored Layout", group);


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

            /* -- Solution coefficients -- */
            // Dimensions of the dataset: (num_elems, nb, ns)
            dimensions.resize(3);
            dimensions[0] = mesh.num_elems_part;
            dimensions[1] = nb;
            dimensions[2] = ns;
            write_dataset(h_Uc.data(), "Solution Coefficients", group,
                dimensions);
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

template <class T>
void Writer::write_attribute(T data, string name, H5::Group group) {
    // Get HDF5 type
    auto type = get_hdf5_type<T>();
    // Create dataspace
    H5::DataSpace dataspace(H5S_SCALAR);
    // Create attribute
    auto attribute = group.createAttribute(name, type, dataspace);
    // Write
    attribute.write(type, (void*)&data);
}

template<>
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

template <>
H5::PredType Writer::get_hdf5_type<unsigned*>() {
    return H5::PredType::NATIVE_UINT;
}

template <>
H5::PredType Writer::get_hdf5_type<bool>() {
    return H5::PredType::NATIVE_HBOOL;
}
