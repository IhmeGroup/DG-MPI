#include <iostream>
#include <sstream>
#include <vector>
#include "io/writer.h"
#include "memory/memory_network.h"
#include "HDF5Wrapper.h"

using std::string, std::vector;

Writer::Writer(Mesh& mesh, MemoryNetwork& network, host_view_type_3D h_Uc,
    int nb, int ns, rtype time) {

    string file_name = "data.h5";

    // Copy mesh info to the host from the device
    mesh.copy_from_device_to_host();

    bool parallel = true;
    HDF5File file(file_name, H5F_ACC_TRUNC, parallel);

    // Write some attributes to the file
    file.create_and_write_attribute("Number of Ranks", network.num_ranks);
    file.create_and_write_attribute("Number of Dimensions", mesh.dim);
    file.create_and_write_attribute("Number of Elements", mesh.num_elems);
    file.create_and_write_attribute("Number of Nodes", mesh.num_nodes);
    file.create_and_write_attribute("Number of Nodes Per Element", mesh.num_nodes_per_elem);
    file.create_and_write_attribute("Number of Basis Functions", nb);
    file.create_and_write_attribute("Number of State Variables", ns);
    file.create_and_write_attribute("Solver Final Time", time);
    file.create_and_write_attribute("Number of Elements per Partition", mesh.num_elems_part);

    bool stored_layout =
        std::is_same_v<decltype(h_Uc)::array_layout,Kokkos::LayoutRight>;
    file.create_and_write_attribute("Stored Layout", stored_layout);

    vector<hsize_t> dimensions;

    /* -- Node coordinates -- */
    // Dimensions of the dataset: (num_nodes, dim)
    dimensions.resize(2);
    dimensions[0] = mesh.num_nodes_part;
    dimensions[1] = mesh.dim;
    file.create_and_write_parallel_dataset("Node Coordinates", dimensions, mesh.h_node_coords.data());

    /* -- Local to global node IDs -- */
    // Dimensions of the dataset: (num_nodes)
    dimensions.resize(1);
    dimensions[0] = mesh.num_nodes_part;
    file.create_and_write_parallel_dataset("Local to Global Node ID", dimensions, mesh.h_local_to_global_node_IDs.data());

    /* -- Element nodes -- */
    // Dimensions of the dataset: (num_elems, num_nodes_per_elem)
    dimensions.resize(2);
    dimensions[0] = mesh.num_elems_part;
    dimensions[1] = mesh.num_nodes_per_elem;
    file.create_and_write_parallel_dataset("Element Global Node IDs", dimensions, mesh.h_elem_to_node_IDs.data());

    /* -- Local to global element IDs -- */
    // Dimensions of the dataset: (num_elems)
    dimensions.resize(1);
    dimensions[0] = mesh.num_elems_part;
    file.create_and_write_parallel_dataset("Local to Global Element IDs", dimensions, mesh.h_local_to_global_elem_IDs.data());

    /* -- Solution coefficients -- */
    // Dimensions of the dataset: (num_elems, nb, ns)
    dimensions.resize(3);
    dimensions[0] = mesh.num_elems_part;
    dimensions[1] = nb;
    dimensions[2] = ns;
    file.create_and_write_parallel_dataset("Solution Coefficients", dimensions, h_Uc.data());
}

