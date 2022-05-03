#ifndef DG_READER_H
#define DG_READER_H

#include "HDF5Wrapper.h"
#include "memory/memory_network.h"
#include "solver/base.h"

class Reader
{
    public:

    template<typename T>
    struct parallel_dataset
    {
        std::string name;

        // rank-local dimensions and rank-local flattened data
        std::vector<T> data;
        std::vector<hsize_t> dimensions;

        hsize_t totalSize;

        // the following data is only available on the head rank
        std::vector<hsize_t> offsets;
        std::vector<int> localSizes; // must be 'int' for compatibility with MPI
    };

    // always read in parallel
    Reader(const std::string& filename, bool serialize_arg=false)
        : file(filename, H5F_ACC_RDWR, true)
    {
        serialize = serialize_arg;
        num_ranks = file.open_and_read_attribute_all<unsigned>("Number of Ranks");

        if (file.get_mpi_size() != num_ranks)
        {
            if (file.get_rank() == HDF5File::head_rank)
            {
                std::cout<<"ERROR: must read file "<<filename<<" with "<<num_ranks<<" MPI rank "<<std::endl;
                std::exit(EXIT_FAILURE);
            }
        }

        dim                = file.open_and_read_attribute_all<unsigned>("Number of Dimensions");
        num_elems          = file.open_and_read_attribute_all<unsigned>("Number of Elements");
        num_nodes          = file.open_and_read_attribute_all<unsigned>("Number of Nodes");
        num_nodes_per_elem = file.open_and_read_attribute_all<unsigned>("Number of Nodes Per Element");
        nb                 = file.open_and_read_attribute_all<unsigned>("Number of Basis Functions");
        ns                 = file.open_and_read_attribute_all<int>("Number of State Variables");
        time               = file.open_and_read_attribute_all<double>("Solver Final Time");
        stored_layout      = file.open_and_read_attribute_all<bool>("Stored Layout");

        num_elems_part     = file.open_and_read_dataset_scatter_scalar<unsigned>("Number of Elements per Partition");

        if (serialize && file.get_mpi_size() > 1)
        {
            unsigned sum = 0;
            MPI_Reduce(&num_elems_part, &sum, 1, MPITYPE<unsigned>, MPI_SUM, HDF5File::head_rank, file.get_comm());
            num_elems_part = sum;
        }

        read_parallel_datasets();

        if (serialize)
        {
            // todo: sorting
        }
    }

    HDF5File file;

    /*
        The following data is available on all ranks
    */

    bool serialize;

    unsigned num_ranks;
    unsigned dim;
    unsigned num_elems;
    unsigned num_nodes;
    unsigned num_nodes_per_elem;
    unsigned nb;
    unsigned ns;
    rtype    time;
    bool     stored_layout;
    unsigned num_elems_part;

    parallel_dataset<rtype>    node_coords;
    parallel_dataset<unsigned> local_to_global_node_IDs;
    parallel_dataset<unsigned> elem_to_node_IDs;
    parallel_dataset<unsigned> local_to_global_elem_IDs;
    parallel_dataset<rtype>    Uc;

    private:

    void read_parallel_datasets()
    {
        populate_parallel_dataset("Node Coordinates", node_coords);
        populate_parallel_dataset("Local to Global Node ID", local_to_global_node_IDs);
        populate_parallel_dataset("Element Global Node IDs", elem_to_node_IDs);
        populate_parallel_dataset("Local to Global Element IDs", local_to_global_elem_IDs);
        populate_parallel_dataset("Solution Coefficients", Uc);
    }

    template<typename T>
    void populate_parallel_dataset(const std::string& name, parallel_dataset<T>& dataset)
    {
        dataset.name = name;
        dataset.dimensions = file.read_dims_of_parallel_dataset(name);
        hsize_t localSize = std::accumulate(std::cbegin(dataset.dimensions), std::cend(dataset.dimensions), static_cast<hsize_t>(1), std::multiplies<hsize_t>());
        dataset.data.resize(localSize);
        dataset.totalSize = file.read_total_size_of_parallel_dataset(name);

        file.open_and_read_parallel_dataset(name, dataset.data.data());

        if (file.get_rank() == HDF5File::head_rank)
        {
            dataset.offsets = file.open_and_read_dataset<hsize_t>(name+"_offsets");
            dataset.localSizes.resize(dataset.offsets.size());
            for (std::size_t i=0; i<dataset.offsets.size()-1; ++i)
                dataset.localSizes[i] = static_cast<int>(dataset.offsets[i+1] - dataset.offsets[i]);
            dataset.localSizes.back() = dataset.totalSize - dataset.offsets.back();
        }

        if (serialize && file.get_mpi_size() > 1)
        {
            std::vector<T> buffer;
            std::vector<hsize_t> dim;
            serialize_dataset(dataset, buffer, dim);
            dataset.data = buffer;
            dataset.dimensions = dim;
        }
    }

    template<typename T>
    void serialize_dataset(parallel_dataset<T>& local_data, std::vector<T>& buffer, std::vector<hsize_t>& dim_out)
    {
        buffer.clear();
        dim_out.clear();

        if (file.get_rank() == HDF5File::head_rank)
        {
            // must be 'int' due to MPI interface
            std::vector<int> offsets_int(local_data.offsets.size());
            for (std::size_t i=0; i!=offsets_int.size(); ++i)
                offsets_int[i] = static_cast<int>(local_data.offsets[i]);

            buffer.resize(local_data.totalSize);

            MPI_Gatherv(local_data.data.data(), local_data.data.size(), MPITYPE<T>, buffer.data(), local_data.localSizes.data(), offsets_int.data(), MPITYPE<T>, HDF5File::head_rank, file.get_comm());

            dim_out.resize(local_data.dimensions.size());
            dim_out[0] = local_data.totalSize;
            for (std::size_t i=1; i<dim_out.size(); ++i)
            {
                dim_out[0] /= local_data.dimensions[i];
                dim_out[i] = local_data.dimensions[i];
            }
        }
        else
            MPI_Gatherv(local_data.data.data(), local_data.data.size(), MPITYPE<T>, nullptr, nullptr, nullptr, MPITYPE<T>, HDF5File::head_rank, file.get_comm());
    }

};

#endif //DG_READER_H

