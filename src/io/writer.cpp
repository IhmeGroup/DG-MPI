#include <iostream>
#include <string>
#include "H5Cpp.h"
#include "io/writer.h"

using std::string;


Writer::Writer(const Mesh& mesh) {
    /* We write one restart file overall. Only point task 0 creates a file. The other point tasks
     * return immediately. The top-level task must wait that point task 0 returned before
     * attempting to write to the restart file. */
    //const string file_name = Utils::get_fname_with_iter(
    //    inputs.restart_file_prefix, ".h5", time_info->iter_curr);
    // TODO: get file name
    const string file_name = string(PROJECT_ROOT) + "/build/test/mpi_enabled_tests/mesh/data.h5";

    H5::H5File file(file_name, H5F_ACC_TRUNC);

    // dimensions of the datasets (1D dataset of size the number of elements)
    hsize_t dim_dset[1];
    dim_dset[0] = mesh.num_nodes;
    // dimensions of datasets' elements (1D array of size the number of solution coefficients)
    hsize_t dim_elem[1];
    dim_elem[0] = mesh.dim;
    // create an array data type
    H5::ArrayType delem_type_real(H5::PredType::NATIVE_DOUBLE, 1, dim_elem);
    // create datasets for the solution coefficients
    H5::DataSpace dataspace(1, dim_dset);
    file.createDataSet("Node Coordinates", delem_type_real, dataspace);

    file.close();
}
