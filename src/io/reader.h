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
        file.open_and_read_attribute_all("Number of Ranks", &num_ranks);

        if (file.mpi_size() != num_ranks)
        {
            if (file.rank() == HDF5File::head_rank)
            {
                std::cout<<"ERROR: must read file "<<filename<<" with "<<num_ranks<<" MPI rank "<<std::endl;
                std::exit(EXIT_FAILURE);
            }
        }

        file.open_and_read_attribute_all("Number of Dimensions", &dim);
        file.open_and_read_attribute_all("Number of Elements", &num_elems);
        file.open_and_read_attribute_all("Number of Nodes", &num_nodes);
        file.open_and_read_attribute_all("Number of Nodes Per Elemen", &num_nodes_per_elem);
        file.open_and_read_attribute_all("Number of Basis Functions", &nb);
        file.open_and_read_attribute_all("Number of State Variables", &ns);
        file.open_and_read_attribute_all("Solver Final Time", &time);
        file.open_and_read_attribute_all("Stored Layout", &stored_layout);

        file.open_and_read_dataset_scatter_scalar("Number of Elements per Partition", &num_elems_part);

        if (serialize && file.mpi_size() > 1)
        {
            unsigned sum = 0;
            MPI_Reduce(&num_elems_part, &sum, 1, MPITYPE<unsigned>, MPI_SUM, HDF5File::read_rank, file.comm());
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
    unsigned num_elems_part

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
        hsize_t localSize = std::accumulate(std::cbegin(dims), std::cend(dims), static_cast<hsize_t>(1), std::multiplies<hsize_t>());
        dataset.data.resize(localSize);
        dataset.totalSize = read_total_size_of_parallel_dataset(name);

        file.open_and_read_parallel_dataset(name, dataset.data.data());

        if (file.rank() == HDF5File::head_rank)
        {
            dataset.offsets = file.open_and_read_dataset(name+"_offsets");
            localSizes.resize(offsets.size());
            for (std::size_t i=0; i<offsets.size()-1; ++i)
                localSizes[i] = static_cast<int>(offsets[i+1] - offsets[i]);
            localSizes.back() = totalSize - offsets.back();
        }

        if (serialize && file.mpi_size() > 1)
        {
            std::vector<T> buffer;
            std::vector<hsize_t> dim;
            serialize_dataset(dataset, buffer, dim);
            dataset.data = buffer;
            dataset.dimension = dim;
        }
    }

    template<typename T>
    void serialize_dataset(const parallel_dataset& local_data, std::vector<T>& buffer, std::vector<hsize_t>& dim_out)
    {
        buffer.clear();
        dim_out.clear();

        if (file.rank() == HDF5File::head_rank)
        {
            // must be 'int' due to MPI interface
            std::vector<int> offsets_int(local_data.offsets.size());
            for (std::size_t i=0; i!=offsets_int.size(); ++i)
                offsets_int[i] = static_cast<int>(local_data.offsets[i]);

            MPI_Gatherv(data_in.data(), data_in.size(), MPITYPE<T>, buffer.data(), localSizes.data(), offsets.size(), MPITYPE<T>, HDF5File::head_rank, file.comm());

            dim_out.resize(dims_in.size());
            dim_out[0] = local_data.totalSize;
            for (std::size_t i=1; i<dim_out.size(); ++i)
            {
                dim_out[0] /= local_data.dimensions[i];
                dim_out[i] = local_data.dimensions[i];
            }
        }
        else
            MPI_Gatherv(data_in.data(), data_in.size(), MPITYPE<T>, nullptr, nullptr, offsets.size(), MPITYPE<T>, HDF5File::head_rank, file.commm());
    }

};

#endif //DG_READER_H

