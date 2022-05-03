#ifndef DG_HDF5FILE_H
#define DG_HDF5FILE_H

#include "hdf5.h"
#include "mpi.h"

#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include <cstdlib>
#include <utility>
#include <numeric>
#include <functional>
#include <tuple>

// requirements for parallel API: https://portal.hdfgroup.org/display/HDF5/Collective+Calling+Requirements+in+Parallel+HDF5+Applications

// todo: dataset read is currently limited to 2 GB of data per rank


template<typename T> hid_t HDF5TYPE();
template<> inline hid_t HDF5TYPE<int>() {return H5T_NATIVE_INT;}
template<> inline hid_t HDF5TYPE<long int>() {return H5T_NATIVE_LONG;}
template<> inline hid_t HDF5TYPE<long long int>() {return H5T_NATIVE_LLONG;}
template<> inline hid_t HDF5TYPE<char>() {return H5T_NATIVE_CHAR;}
template<> inline hid_t HDF5TYPE<unsigned int>() {return H5T_NATIVE_UINT;}
template<> inline hid_t HDF5TYPE<unsigned long int>() {return H5T_NATIVE_ULONG;}
template<> inline hid_t HDF5TYPE<unsigned long long int>() {return H5T_NATIVE_ULLONG;}
template<> inline hid_t HDF5TYPE<float>() {return H5T_NATIVE_FLOAT;}
template<> inline hid_t HDF5TYPE<double>() {return H5T_NATIVE_DOUBLE;}

using mpitype = decltype(MPI_DATATYPE_NULL);
template<typename T> constexpr mpitype MPITYPE;
template<> constexpr mpitype MPITYPE<int> = MPI_INT;
template<> constexpr mpitype MPITYPE<char> = MPI_CHAR;
template<> constexpr mpitype MPITYPE<unsigned int> = MPI_UNSIGNED;
template<> constexpr mpitype MPITYPE<long long int> = MPI_LONG_LONG;
template<> constexpr mpitype MPITYPE<long int> = MPI_LONG;
template<> constexpr mpitype MPITYPE<unsigned long int> = MPI_UNSIGNED_LONG;
template<> constexpr mpitype MPITYPE<unsigned long long int> = MPI_UNSIGNED_LONG_LONG;
template<> constexpr mpitype MPITYPE<float> = MPI_FLOAT;
template<> constexpr mpitype MPITYPE<double> = MPI_DOUBLE;

class HDF5File
{
    using managed_hid = std::shared_ptr<std::pair<hid_t,hid_t>>;


    public:

        static constexpr int head_rank = 0;

        HDF5File() : file(-1), parallel(false), comm(MPI_COMM_WORLD), info(MPI_INFO_NULL), mpi_size(1), mpi_rank(head_rank) {}
        HDF5File(const HDF5File&) = delete;
        HDF5File(HDF5File&& rhs) :
            file(rhs.file), parallel(rhs.parallel), comm(rhs.comm), info(rhs.info), mpi_size(rhs.mpi_size), mpi_rank(rhs.mpi_rank)
        {
            rhs.file = -1;
        }
        HDF5File& operator=(HDF5File&& rhs)
        {
            if (&rhs == this)
                return *this;

            file = rhs.file;
            rhs.file = -1;
            parallel = rhs.parallel;
            comm = rhs.comm;
            info = rhs.info;
            mpi_size = rhs.mpi_size;
            mpi_rank = rhs.mpi_rank;
            return *this;
        }
        HDF5File& operator=(const HDF5File& rhs) = delete;
        HDF5File(const std::string& name, unsigned mode, bool par=false, MPI_Comm comm=MPI_COMM_WORLD, MPI_Info info=MPI_INFO_NULL)
            : file(-1), parallel(par), mpi_size(1), mpi_rank(head_rank)
        {
            open(name, mode, par, comm, info);
        }
        ~HDF5File() {close();}

        // if used in parallel: must be called collectively with the the same arguments on each rank
        void open(const std::string& name, unsigned mode, bool par=false, MPI_Comm Comm=MPI_COMM_WORLD, MPI_Info Info=MPI_INFO_NULL)
        {
            close();

            parallel = par;

            if (parallel)
            {
                comm = Comm;
                info = Info;
                MPI_Comm_size(comm, &mpi_size);
                MPI_Comm_rank(comm, &mpi_rank);
            }
            else
            {
                mpi_size = 1;
                mpi_rank = head_rank;
            }

            hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);
            if (plist_id < 0)
            {
                std::cerr<<"ERROR: cannot call H5Pcreate"<<std::endl;
                std::exit(EXIT_FAILURE);
            }

            if (parallel)
            {
                if (H5Pset_fapl_mpio(plist_id, comm, info) < 0)
                {
                    std::cerr<<"ERROR: cannot call H5Pset_fapl_mpio"<<std::endl;
                    std::exit(EXIT_FAILURE);
                }
            }

            if (mode == H5F_ACC_RDONLY || mode == H5F_ACC_RDWR)
            {
                file = H5Fopen(name.c_str(), mode, plist_id);
                if (file < 0)
                {
                    std::cerr<<"ERROR: cannot call H5Fopen for "<<name<<std::endl;
                    std::exit(EXIT_FAILURE);
                }
            }
            else if(mode == H5F_ACC_TRUNC || mode == H5F_ACC_EXCL )
            {
                file = H5Fcreate(name.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
                if (file < 0)
                {
                    std::cerr<<"ERROR: cannot call H5Fcreate for "<<name<<std::endl;
                    std::exit(EXIT_FAILURE);
                }
            }
            else
            {
                std::cerr<<"ERROR: invalid mode for "<<name<<std::endl;
                std::exit(EXIT_FAILURE);
            }

            if (H5Pclose(plist_id) < 0)
            {
                std::cerr<<"ERROR: cannot call H5Pclose"<<std::endl;
                std::exit(EXIT_FAILURE);
            }
        }

        // if used in parallel: must be called collectively with the the same arguments on each rank
        void close()
        {
            if (file != -1)
            {
                if(H5Fclose(file) < 0)
                {
                    std::cerr<<"ERROR: cannot close file"<<std::endl;
                    std::exit(EXIT_FAILURE);
                }
                file = -1;
            }
        }

        // if used in parallel mode, must be called by all ranks with the same parameter i.e. name (currently not enforcd by the API)
        auto create_group(const std::string& name, managed_hid destination = managed_hid())
        {
            managed_hid p = new_group();

            p->second = H5Gcreate(get_dest(destination), name.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            if (p->second < 0)
            {
                std::cerr<<"ERROR: cannot create group "<<name<<std::endl;
                std::exit(EXIT_FAILURE);
            }
            return p;
        }


        // if used in parallel: if the group is never modified / written to, can be called independently
        auto open_group(const std::string& name, managed_hid destination = managed_hid())
        {
            managed_hid p = new_group();

            p->second = H5Gopen(get_dest(destination), name.c_str(), H5P_DEFAULT);
            if (p->second < 0)
            {
                std::cerr<<"ERROR: cannot open group "<<name<<std::endl;
                std::exit(EXIT_FAILURE);
            }
            return p;
        }

        // if used in parallel: can be called independently
        bool link_exists(const std::string& name, managed_hid destination = managed_hid())
        {
            return H5Lexists(get_dest(destination), name.c_str(), H5P_DEFAULT) > 0;
        }

        // must be called by all ranks
        template<typename T>
        auto create_attribute(const std::string& name, managed_hid destination = managed_hid())
        {
            managed_hid p = new_attribute();

            p->first = H5Screate(H5S_SCALAR);
            if (p->first < 0)
            {
                std::cerr<<"ERROR: cannot create data space for "<<name<<std::endl;
                std::exit(EXIT_FAILURE);
            }

            hid_t type;
            if constexpr (std::is_same_v<T,bool>)
                type = HDF5TYPE<int>();
            else
                type = HDF5TYPE<T>();


            p->second = H5Acreate(get_dest(destination), name.c_str(), type, p->first, H5P_DEFAULT, H5P_DEFAULT);
            if (p->second < 0)
            {
                std::cerr<<"ERROR cannot create attribute for "<<name<<std::endl;
                std::exit(EXIT_FAILURE);
            }

            return p;
        }

        // if used in parallel mode, can be used independently, if the attribute is never modified afterwards
        template<typename T>
        auto open_attribute(const std::string& name, managed_hid destination = managed_hid())
        {
            managed_hid p = new_attribute();

            p->second = H5Aopen_by_name(get_dest(destination), ".", name.c_str(), H5P_DEFAULT, H5P_DEFAULT);
            if (p->second < 0)
            {
                std::cerr<<"ERROR cannot open attribute "<<name<<std::endl;
                std::exit(EXIT_FAILURE);
            }

            return p;
        }

        // can be called independently. If the attribute should be available on all ranks, better call read_attribute_all
        template<typename T>
        T read_attribute(managed_hid& att)
        {
            T value;
            herr_t err = -1;
            if constexpr(std::is_same_v<T,bool>) // emulate bool with int
            {
                int tmp;
                err = H5Aread(att->second, HDF5TYPE<int>(), &tmp);
                value = tmp != 0;
            }
            else
                err = H5Aread(att->second, HDF5TYPE<T>(), &value);

            if (err < 0)
            {
                std::cerr<<"ERROR: cannot read from attribute"<<std::endl;
                std::exit(EXIT_FAILURE);
            }

            return value;
        }


        // if used in parallel mode, must be called by all ranks. Only the head rank will read and then broadcast the results to the other ranks
        template<typename T>
        T read_attribute_all(managed_hid& att)
        {
            T value;

            if (parallel && mpi_rank != head_rank)
            {
                    if constexpr(std::is_same_v<T,bool>) // emulate bool with int
                    {
                        int tmp = -1;
                        MPI_Bcast(&tmp, 1, MPITYPE<int>, head_rank, comm);
                        value = tmp != 0;
                    }
                    else
                        MPI_Bcast(&value, 1, MPITYPE<T>, head_rank, comm);
            }
            else
            {
                value = read_attribute<T>(att);
                if (parallel)
                {
                    if constexpr(std::is_same_v<T,bool>) // emulate bool with int
                    {
                        int tmp = value;
                        MPI_Bcast(&tmp, 1, MPITYPE<int>, head_rank, comm);
                        value = tmp != 0;
                    }
                    else
                        MPI_Bcast(&value, 1, MPITYPE<T>, head_rank, comm);

                }
            }
            return value;
        }

        // can be called independently
        template<typename T>
        T open_and_read_attribute(const std::string& name, managed_hid destination = managed_hid())
        {
            auto att = open_attribute<T>(name, destination);
            return read_attribute<T>(att);
        }

        // must be called collectively
        template<typename T>
        T open_and_read_attribute_all(const std::string& name, managed_hid destination = managed_hid())
        {
            managed_hid att;
            if (!parallel || mpi_rank == head_rank)
                att = open_attribute<T>(name, destination);
            return read_attribute_all<T>(att);
        }



        // if used in parallel mode, must (apparently?) be called by all ranks. Should be called with same 'value' on all ranks (not enforced by at the moment)
        template<typename T>
        auto write_to_attribute(managed_hid& att, const T& value)
        {
            herr_t err = -1;
            if constexpr(std::is_same_v<T,bool>) // emulate bool with int
            {
                int tmp = value != 0;
                err = H5Awrite(att->second, HDF5TYPE<int>(), &tmp);
            }
            else
                err = H5Awrite(att->second, HDF5TYPE<T>(), &value);

            if (err < 0)
            {
                std::cerr<<"ERROR: cannot write to attribute"<<std::endl;
                std::exit(EXIT_FAILURE);
            }
        }

        // if used in parallel mode, must (apparently?) be called by all ranks. Should be called with same 'value' on all ranks (not enforced by at the moment)
        template<typename T>
        auto create_and_write_attribute(const std::string& name, const T& value, managed_hid destination = managed_hid())
        {
            auto att = create_attribute<T>(name, destination);
            write_to_attribute(att, value);
        }

        // if used in parallel mode, all ranks must call this. The 'dims' argument of all ranks other than the head rank will be ignored.
        // name should be the same on all ranks (currently not enforced)
        // if all ranks are supposed to write different data to a dataset, use 'create_dataset_prallel' instead
        template<typename T>
        auto create_dataset(const std::string& name, const std::vector<hsize_t>& dims, managed_hid destination = managed_hid())
        {
                auto p = new_dataset();

                //make sure, dims is the same for all ranks
                if (parallel)
                {
                    std::vector<hsize_t> dims_parallel(dims);
                    if (mpi_rank == head_rank)
                    {
                        decltype(dims_parallel)::size_type length = dims.size();
                        MPI_Bcast(&length, 1, MPITYPE<hsize_t>, head_rank, comm);
                        MPI_Bcast(dims_parallel.data(), length, MPITYPE<hsize_t>, head_rank, comm);
                    }
                    else
                    {
                        decltype(dims_parallel)::size_type length;
                        MPI_Bcast(&length, 1, MPITYPE<hsize_t>, head_rank, comm);
                        dims_parallel.resize(length);
                        MPI_Bcast(dims_parallel.data(), length, MPITYPE<hsize_t>, head_rank, comm);
                    }
                    p->first = H5Screate_simple(dims_parallel.size(), dims_parallel.data(), dims_parallel.data());
                }
                else
                    p->first = H5Screate_simple(dims.size(), dims.data(), dims.data());

                if (p->first < 0)
                {
                    std::cerr<<"ERROR: cannot create space"<<std::endl;
                    std::exit(EXIT_FAILURE);
                }

                p->second = H5Dcreate(get_dest(destination), name.c_str(), HDF5TYPE<T>(), p->first, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
                if (p->second < 0)
                {
                    std::cerr<<"ERROR: cannot create dataset"<<std::endl;
                    std::exit(EXIT_FAILURE);
                }
                return p;
        }

        // special case for one-dimensional dataset
        // must be called by all ranks. The 'length' parameter is ignored for all ranks except for the head rank 
        template<typename T>
        managed_hid create_dataset(const std::string& name, hsize_t length, managed_hid destination = managed_hid())
        {
            return create_dataset<T>(name, std::vector<hsize_t>{length}, destination);
        }

        // must be called by all ranks if the dataset is later modified
        auto open_dataset(const std::string& name, managed_hid destination = managed_hid())
        {
            auto p = new_dataset();

            p->second = H5Dopen(get_dest(destination), name.c_str(), H5P_DEFAULT);
            if (p->second < 0)
            {
                std::cerr<<"ERROR: cannot create dataset"<<std::endl;
                std::exit(EXIT_FAILURE);
            }
            p->first = H5Dget_space(p->second);
            return p;
        }

        // if used in parallel mode: can be called independently.
        // If the dimensions should be available on all ranks, call 'dataset_dimensions_all' instead
        std::vector<hsize_t> dataset_dimensions(const managed_hid& dset)
        {
            auto nDims = H5Sget_simple_extent_ndims(dset->first);
            if (nDims < 0)
            {
                std::cerr<<"ERROR: cannot get number of dimensions"<<std::endl;
                std::exit(EXIT_FAILURE);
            }

            std::vector<hsize_t> dims(nDims, -1);
            if (H5Sget_simple_extent_dims(dset->first, dims.data(), nullptr) < 0)
            {
                std::cerr<<"ERROR: cannot get dimensions"<<std::endl;
                std::exit(EXIT_FAILURE);
            }
            return dims;
        }

        // if used in parallel mode, must be called by all ranks
        std::vector<hsize_t> dataset_dimensions_all(const managed_hid& dset)
        {
            std::vector<hsize_t> dims;

            if (parallel && mpi_rank != head_rank)
            {
                hsize_t length = -1;
                MPI_Bcast(&length, 1, MPITYPE<hsize_t>, head_rank, comm);
                dims.resize(length);
                MPI_Bcast(dims.data(), length, MPITYPE<hsize_t>, head_rank, comm);

            }
            else
            {
                dims = dataset_dimensions(dset);
                if (parallel)
                {
                    hsize_t length = dims.size();
                    MPI_Bcast(&length, 1, MPITYPE<hsize_t>, head_rank, comm);
                    MPI_Bcast(dims.data(), length, MPITYPE<hsize_t>, head_rank, comm);
                }
            }

            return dims;
        }

        // if used in parallel mode: can be called independently.
        // If the dataset should be available on all ranks, call 'dataset_dimensions_all' instead
        template<typename T>
        void read_dataset(const managed_hid& dset, T* data)
        {
            if (H5Dread(dset->second, HDF5TYPE<T>(), H5S_ALL, H5S_ALL, H5P_DEFAULT, data) < 0)
            {
                std::cerr<<"ERROR: cannot read dataset"<<std::endl;
                std::exit(EXIT_FAILURE);
            }
        }

        // if used in parallel mode, must be called by all ranks. 'data' must be large enough in all ranks to contain the whole dataset
        template<typename T>
        void read_dataset_all(const managed_hid& dset, T* data)
        {
            auto dims = dataset_dimensions_all(dset);
            hsize_t nElements = std::accumulate(std::cbegin(dims), std::cend(dims), static_cast<hsize_t>(1), std::multiplies<hsize_t>());

            if (parallel && mpi_rank != head_rank)
                MPI_Bcast(data, nElements, MPITYPE<T>, head_rank, comm);
            else
            {
                read_dataset(dset, data);

                if (parallel)
                    MPI_Bcast(data, nElements, MPITYPE<T>, head_rank, comm);
            }
        }



        // can be called independently. If the dataset should be available on all ranks, better call read_dataset_all
        template<typename T>
        std::vector<T> read_dataset(const managed_hid& dset)
        {
            auto dims = dataset_dimensions(dset);

            hsize_t nElements = std::accumulate(std::cbegin(dims), std::cend(dims), static_cast<hsize_t>(1), std::multiplies<hsize_t>());

            std::vector<T> v(nElements);

            read_dataset(dset, v.data());

            return v;
        }

        // must be called by all ranks
        template<typename T>
        std::vector<T> read_dataset_all(const managed_hid& dset)
        {
            // todo: dataset_dimensions_all includes a broadcast of dimensions, which is also done in read_dataset_all. Could be avoided later
            auto dims = dataset_dimensions_all(dset);

            hsize_t nElements = std::accumulate(std::cbegin(dims), std::cend(dims), static_cast<hsize_t>(1), std::multiplies<hsize_t>());

            std::vector<T> v(nElements);

            read_dataset_all(dset, v.data());

            return v;
        }


        // can be called independently. If the data should be read to all ranks, call 'open_and_read_dataset_all' instead
        template<typename T>
        void open_and_read_dataset(const std::string& name, T* data,  managed_hid destination = managed_hid())
        {
            auto dset = open_dataset(name, destination);
            read_dataset<T>(dset, data);
        }

        // must ba called by all ranks 
        template<typename T>
        void open_and_read_dataset_all(const std::string& name, T* data,  managed_hid destination = managed_hid())
        {
            managed_hid dset;
            if (!parallel || mpi_rank==head_rank)
                dset = open_dataset(name, destination);
            read_dataset_all<T>(dset, data);
        }

        // can be called independently. If the data should be read to all ranks, call 'open_and_read_dataset_all' instead
        template<typename T>
        std::vector<T> open_and_read_dataset(const std::string& name, managed_hid destination = managed_hid())
        {
            auto dset = open_dataset(name, destination);
            return read_dataset<T>(dset);
        }

        // must be called by all ranks
        template<typename T>
        std::vector<T> open_and_read_dataset_all(const std::string& name, managed_hid destination = managed_hid())
        {
            managed_hid dset;
            if (!parallel || mpi_rank==head_rank)
                dset = open_dataset(name, destination);
            return read_dataset_all<T>(dset);
        }

        // can be called independently. If the data and dims should be read to all ranks, call 'open_and_read_dataset_all' instead
        template<typename T>
        std::tuple<std::vector<T>, std::vector<hsize_t>> open_and_read_dataset_with_dims(const std::string& name, managed_hid destination = managed_hid())
        {
            auto dset = open_dataset(name, destination);
            return {read_dataset<T>(dset), dataset_dimensions(dset)};
        }

        // must be called by all ranks
        template<typename T>
        std::tuple<std::vector<T>, std::vector<hsize_t>> open_and_read_dataset_with_dims_all(const std::string& name, managed_hid destination = managed_hid())
        {
            managed_hid dset;
            if (!parallel || mpi_rank==head_rank)
                dset = open_dataset(name, destination);
            return {read_dataset_all<T>(dset), dataset_dimensions_all(dset)};
        }

        // can be called independently, if the dataset and dims should be available on all ranks, call 'open_and_read_dataset_with_dims_all'
        template<typename T>
        auto open_and_read_dataset_with_dims(const std::string& name, T* data, managed_hid destination = managed_hid())
        {
            auto dset = open_dataset(name, destination);
            read_dataset<T>(dset, data);
            return dataset_dimensions(dset);
        }

        // must be called by all ranks
        template<typename T>
        auto open_and_read_dataset_with_dims_all(const std::string& name, T* data, managed_hid destination = managed_hid())
        {
            managed_hid dset;
            if (!parallel || mpi_rank==head_rank)
                dset = open_dataset(name, destination);
            read_dataset_all<T>(dset, data);
            return dataset_dimensions_all(dset);
        }

        // can be called independently
        template<typename T>
        void write_to_dataset(const managed_hid& dset, const T* data)
        {
            if (H5Dwrite(dset->second, HDF5TYPE<T>(), H5S_ALL, H5S_ALL, H5P_DEFAULT, data) < 0)
            {
                std::cerr<<"ERROR: cannot write to dataset"<<std::endl;
                std::exit(EXIT_FAILURE);

            }
        }

        // must be called by all ranks. only the data on the head rank will be written
        template<typename T>
        void create_and_write_dataset(const std::string& name, const std::vector<hsize_t>& dims, const T* data, managed_hid destination = managed_hid())
        {
            auto dset = create_dataset<T>(name, dims, destination);
            if (!parallel || mpi_rank == head_rank)
                write_to_dataset(dset, data);
        }

        // must be called by all ranks. only the data on the head rank will be written
        template<typename T>
        void create_and_write_dataset(const std::string& name, hsize_t length, const T* data, managed_hid destination = managed_hid())
        {
            auto dset = create_dataset<T>(name, length, destination);
            if (!parallel || mpi_rank == head_rank)
                write_to_dataset(dset, data);
        }

        // must be called by all ranks. each rank passes one value and the head rank writes a dataset composed of these values
        template<typename T>
        void create_and_write_dataset_gather_scalar(const std::string& name, T value, managed_hid destination = managed_hid())
        {
            std::vector<T> dataset;
            if (!parallel || mpi_rank == head_rank)
            {
                dataset.resize(mpi_size);
                dataset[mpi_rank] = value;
            }
            if (parallel)
            {
                if (mpi_rank == head_rank)
                    MPI_Gather(&value, 1, MPITYPE<T>, dataset.data(), 1, MPITYPE<T>, head_rank, comm);
                else
                    MPI_Gather(&value, 1, MPITYPE<T>, nullptr, 1, MPITYPE<T>, head_rank, comm);
            }
            create_and_write_dataset(name, dataset.size(), dataset.data(), destination);
        }

        // must be called by all ranks. returns the nth element from the dataset on rank n
        template<typename T>
        T open_and_read_dataset_scatter_scalar(const std::string& name, managed_hid destination = managed_hid())
        {
            std::vector<T> data;
            T value;
            if (!parallel || mpi_rank == head_rank)
            {
                data = open_and_read_dataset<T>(name, destination);
                value = data[0];
            }

            if (parallel)
            {
                if (mpi_rank == head_rank)
                    MPI_Scatter(data.data(), 1, MPITYPE<T>, &value, 1, MPITYPE<T>, head_rank, comm);
                else
                    MPI_Scatter(nullptr, 1, MPITYPE<T>, &value, 1, MPITYPE<T>, head_rank, comm);
            }

            return value;
        }


        /*
                Parallel datasets:
                For each parallel dataset, a single hdf5 dataset is created
                this is a flattened version of whatever the input data is

                for each dataset 'name', an additional dataset 'name_offsets' is created with the length of mpi_size, containing the offsets of each ranks
                the dataset 'name_offsets' also has two attributes
                 - nDims: the number dimensions of the original input data (before flattening)
                 - totalSize: the total number of elements across all ranks
                 - nRanks - the number of ranks that wrote the dataset

                Also, a dataset 'name_dims' is created, that contains the dimensions of each data block. If e.g. nDims = 2, mpi_size=3, 'name_dims' might contain:
                [ 2 4    2 5    3 4]
                The first two elements are the first and second dimension for the data block of the first rank and so on
        
                Currently, reading a dataset must be done with the same number of ranks as it was written
        
        */

        template<typename T>
        std::tuple<std::vector<T>, std::vector<hsize_t>> open_and_read_parallel_dataset(const std::string& name, managed_hid destination = managed_hid())
        {
            auto dims = read_dims_of_parallel_dataset(name, destination);
            hsize_t localSize = std::accumulate(std::cbegin(dims), std::cend(dims), static_cast<hsize_t>(1), std::multiplies<hsize_t>());
            std::vector<T> data(localSize);
            open_and_read_parallel_dataset(name, data.data(), destination);
            return {data, dims};
        }


        template<typename T>
        auto open_and_read_parallel_dataset(const std::string& name, T* data, managed_hid destination = managed_hid())
        {
            auto dims = read_dims_of_parallel_dataset(name, destination);
            hsize_t localSize = std::accumulate(std::cbegin(dims), std::cend(dims), static_cast<hsize_t>(1), std::multiplies<hsize_t>());
            hsize_t totalSize = read_total_size_of_parallel_dataset(name, destination);
            hsize_t offset = read_offset_of_parallel_dataset(name, destination);

            hid_t dset_id = H5Dopen(get_dest(destination), name.c_str(), H5P_DEFAULT);

            if (dset_id<0)
            {
                std::cerr<<"ERROR: cannot call H5Dopen"<<std::endl;
                std::exit(EXIT_FAILURE);
            }

            hid_t dataspace = H5Screate_simple(1, &totalSize, &totalSize);
            if (dataspace<0)
            {
                std::cerr<<"ERROR: cannot call H5Dget_space"<<std::endl;
                std::exit(EXIT_FAILURE);
            }

            hsize_t one = 1;
            hid_t mspace = H5Screate_simple(1, &localSize, &localSize);
            if (mspace < 0 )
            {
                std::cerr<<"ERROR: cannot call create memspace"<<std::endl;
                std::exit(EXIT_FAILURE);
            }
            if (H5Sselect_all(mspace) < 0)
            {
                std::cerr<<"ERROR: cannot call H5Sselect_all(mspace)"<<std::endl;
                std::exit(EXIT_FAILURE);
            }

            if (H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, &offset, NULL, &one, &localSize) <  0)
            {
                std::cerr<<"ERROR: cannot call H5Sselect_hyperslab"<<std::endl;
                std::exit(EXIT_FAILURE);
            }
            if (H5Dread(dset_id, HDF5TYPE<T>(), mspace, dataspace, H5P_DEFAULT, data) < 0)
            {
                std::cerr<<"ERROR: cannot call H5Dread"<<std::endl;
                std::exit(EXIT_FAILURE);
            }
            if (H5Dclose (dset_id))
            {
                std::cerr<<"ERROR: cannot call H5Dclose"<<std::endl;
                std::exit(EXIT_FAILURE);
            }

            if (H5Sclose(dataspace) < 0)
            {
                std::cerr<<"ERROR: cannot call H5Sclose"<<std::endl;
                std::exit(EXIT_FAILURE);
            }

            if (H5Sclose(mspace) < 0)
            {
                std::cerr<<"ERROR: cannot call  H5Sclose (mspace)"<<std::endl;
                std::exit(EXIT_FAILURE);
            }

            return dims;
        }


        // must be called by all ranks
        template<typename T>
        void create_and_write_parallel_dataset(const std::string& name, const std::vector<hsize_t>& dims, const T* data, managed_hid destination = managed_hid())
        {
            // ====== 1) create the dataspace and write the local data block in parallel ======

            hsize_t localSize = std::accumulate(std::cbegin(dims), std::cend(dims), static_cast<hsize_t>(1), std::multiplies<hsize_t>());
            hsize_t totalSize = 0;
            if (parallel)
                MPI_Allreduce(&localSize, &totalSize, 1, MPI_UINT64_T, MPI_SUM, comm);
            else
                totalSize = localSize;

            hid_t filespace = H5Screate_simple(1, &totalSize, &totalSize);
            if (filespace < 0)
            {
                std::cerr<<"ERROR: cannot create filespace"<<std::endl;
                std::exit(EXIT_FAILURE);
            }

            hid_t dset_id = H5Dcreate(get_dest(destination), name.c_str(), HDF5TYPE<T>(), filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            if (dset_id < 0)
            {
                std::cerr<<"ERROR: cannot create dataspace"<<std::endl;
                std::exit(EXIT_FAILURE);

            }

            hsize_t offset = 0;
            if (parallel)
            {
                MPI_Scan(&localSize, &offset, 1, MPI_UINT64_T, MPI_SUM, comm);
                offset -= localSize;
            }

            hid_t mspace = H5Screate_simple(1, &localSize, &localSize);
            if (mspace < 0 )
            {
                std::cerr<<"ERROR: cannot create memspace"<<std::endl;
                std::exit(EXIT_FAILURE);
            }
            if (H5Sselect_all(mspace) < 0)
            {
                std::cerr<<"ERROR: cannot call H5Sselect_all(mspace)"<<std::endl;
                std::exit(EXIT_FAILURE);
            }

            hsize_t one = 1;
            if (H5Sselect_hyperslab(filespace, H5S_SELECT_SET, &offset, NULL, &one, &localSize) < 0)
            {
                std::cerr<<"ERROR: cannot create hyperslab"<<std::endl;
                std::exit(EXIT_FAILURE);
            }

            if (H5Dwrite(dset_id, HDF5TYPE<T>(), mspace, filespace, H5P_DEFAULT, data) < 0)
            {
                std::cerr<<"ERROR: cannot write hyperslab"<<std::endl;
                std::exit(EXIT_FAILURE);
            }

            if (H5Sclose(mspace) < 0) { std::cerr<<"ERROR: H5Sclose(mspace)"<<std::endl; std::exit(EXIT_FAILURE);}
            if (H5Dclose(dset_id) < 0) { std::cerr<<"ERROR: H5Dclose(dset_id)"<<std::endl; std::exit(EXIT_FAILURE);}
            if (H5Sclose(filespace) < 0) { std::cerr<<"ERROR: H5Sclose(filespace)"<<std::endl; std::exit(EXIT_FAILURE);}


            // ====== 2) write the offset information  ======

            std::vector<hsize_t> offsets;
            if (parallel)
            {
                if (mpi_rank == head_rank)
                {
                    offsets.resize(mpi_size);
                    MPI_Gather(&offset, 1, MPITYPE<hsize_t>, offsets.data(), 1, MPITYPE<hsize_t>, head_rank, comm);
                }
                else
                    MPI_Gather(&offset, 1, MPITYPE<hsize_t>, nullptr, 1, MPITYPE<hsize_t>, head_rank, comm);
            }
            else
                offsets = {offset};

            std::string offset_name = name+"_offsets";

            auto d = create_dataset<hsize_t>(offset_name, static_cast<hsize_t>(mpi_size), destination);
            if (!parallel || mpi_rank == head_rank)
                write_to_dataset(d, offsets.data());

            // ====== 3) write the dimension information  ======

            auto nDims = dims.size();

            if (nDims == 0)
                nDims = 1;

            create_and_write_attribute("nDims", nDims, d);
            create_and_write_attribute("nRanks", mpi_size, d);
            create_and_write_attribute("totalSize", totalSize, d);

            std::vector<hsize_t> dims_all;
            if (parallel)
            {
                if (mpi_rank == head_rank)
                {
                    dims_all.resize(nDims*mpi_size);
                    MPI_Gather(dims.data(), nDims, MPITYPE<hsize_t>, dims_all.data(), nDims, MPITYPE<hsize_t>, head_rank, comm);
                }
                else
                    MPI_Gather(dims.data(), nDims, MPITYPE<hsize_t>, nullptr, nDims, MPITYPE<hsize_t>, head_rank, comm);
            }
            else
                dims_all = dims;

            std::string dim_name = name+"_dims";
            create_and_write_dataset(dim_name, dims_all.size(), dims_all.data(), destination);
        }

        // must be called by all ranks. convenience function for one-dimensional data
        template<typename T>
        void create_and_write_parallel_dataset(const std::string& name, hsize_t length, const T* data, managed_hid destination = managed_hid())
        {
            std::vector<hsize_t> dims{length};
            create_and_write_parallel_dataset(name, dims, data, destination);
        }

        auto read_nRanks_of_parallel_dataset(const std::string& name, managed_hid destination = managed_hid())
        {
            auto d = open_dataset(name+"_offsets", destination);
            return open_and_read_attribute_all<int>("nRanks", d);
        }

        // read the dimensions of the local data block in the parallel dataset. must be called by all ranks
        auto read_dims_of_parallel_dataset(const std::string& name, managed_hid destination = managed_hid())
        {

            // check if the number or ranks reading the file is the same as the number of ranks that wrote the file
            // since this function is called first in open_and_read_parallel_dataset. it is sufficient to do the check here

            auto d = open_dataset(name+"_offsets", destination);
            auto nDim = open_and_read_attribute_all<hsize_t>("nDims", d);
            auto nRanks = open_and_read_attribute_all<int>("nRanks", d);
            bool validCall = true;
            if (parallel)
                validCall = nRanks == mpi_size;
            else
                validCall = nRanks == 1;
            if (!validCall)
            {
                std::cerr<<"ERROR: parallel datasets must be read with the same number of ranks it was written from"<<std::endl;
                std::exit(EXIT_FAILURE);
            }

            std::vector<hsize_t> dims(nDim);
            if (parallel)
            {
                if (mpi_rank == head_rank)
                {
                    auto v = open_and_read_dataset<hsize_t>(name+"_dims", destination);
                    MPI_Scatter(v.data(), nDim, MPITYPE<hsize_t>, dims.data(), nDim, MPITYPE<hsize_t>, head_rank, comm);
                }
                else
                    MPI_Scatter(nullptr, nDim, MPITYPE<hsize_t>, dims.data(), nDim, MPITYPE<hsize_t>, head_rank, comm);
            }
            else
                dims = open_and_read_dataset<hsize_t>(name+"_dims", destination);

            return dims;
        }

        // must be called by all ranks. returns the local number of elements in the parallel datasets for each local data blocks
        auto read_local_size_of_parallel_dataset(const std::string& name, managed_hid destination = managed_hid())
        {
            auto dims = read_dims_of_parallel_dataset(name, destination);
            return std::accumulate(std::cbegin(dims), std::cend(dims), static_cast<hsize_t>(1), std::multiplies<hsize_t>());
        }

        // must be called by all ranks. returns a list of all rank local sizes 
        auto read_all_local_sizes_of_parallel_dataset(const std::string& name, managed_hid destination = managed_hid())
        {
            std::vector<hsize_t> sizes(mpi_size);
            auto localSize = read_local_size_of_parallel_dataset(name, destination);
            sizes[mpi_rank] = localSize;
            if (parallel)
            {
                MPI_Allgather(&localSize, 1, MPITYPE<hsize_t>, sizes.data(), 1, MPITYPE<hsize_t>, comm);
            }
            return sizes;
        }



        // must be called by all ranks. returns the total number of elements in the parallel datasets across all local data blocks
        auto read_total_size_of_parallel_dataset(const std::string& name, managed_hid destination = managed_hid())
        {
            auto d = open_dataset(name+"_offsets", destination);
            return open_and_read_attribute_all<hsize_t>("totalSize", d);
        }

        // return the list of offsets for each rank. must be called by all ranks
        auto read_all_offsets_of_parallel_dataset(const std::string& name, managed_hid destination = managed_hid())
        {
            return open_and_read_dataset_all<hsize_t>(name+"_offsets", destination);
        }

        // must be called by each rank. returns the offset for each individual rank
        auto read_offset_of_parallel_dataset(const std::string& name, managed_hid destination = managed_hid())
        {
            hsize_t offset = 0;
            if (parallel)
            {
                if (mpi_rank == head_rank)
                {
                    auto v = open_and_read_dataset<hsize_t>(name+"_offsets", destination);
                    MPI_Scatter(v.data(), 1, MPITYPE<hsize_t>, &offset, 1, MPITYPE<hsize_t>, head_rank, comm);
                }
                else
                    MPI_Scatter(nullptr, 1, MPITYPE<hsize_t>, &offset, 1, MPITYPE<hsize_t>, head_rank, comm);
            }
            return offset;
        }

        int get_rank()     const { return mpi_rank; }
        int get_mpi_size() const { return mpi_size; }
        MPI_Comm get_comm() const { return comm; }

    private:
        hid_t file;
        bool parallel;
        MPI_Comm comm;
        MPI_Info info;
        int mpi_size;
        int mpi_rank;



    // convenience wrappers for automatically closing handles from hdf5
    // managed_hid.first is an optional handle to a dataspace
    // mangaed_hid.second is a handle to a group, attribute or dataset

    managed_hid new_attribute() const
    {
        return managed_hid(new std::pair<hid_t,hid_t>{-1,-1},
           [](std::pair<hid_t,hid_t>* p)
            {
                if (p == nullptr)
                    return;
                if (p->second != -1 && H5Aclose(p->second) < 0)
                {
                    std::cerr<<"ERROR: cannot close attribute"<<std::endl;
                    std::exit(EXIT_FAILURE);
                }
                if (p->first != -1 && H5Sclose(p->first) < 0)
                {
                    std::cerr<<"ERROR: cannot close dataspace"<<std::endl;
                    std::exit(EXIT_FAILURE);
                }

                delete p;
                p = nullptr;
            });
    }
    managed_hid new_dataset() const
    {
        return managed_hid(new std::pair<hid_t,hid_t>{-1,-1},
            [](std::pair<hid_t,hid_t>* p)
            {
                if (p == nullptr)
                    return;
                if (p->second != -1 && H5Dclose(p->second) < 0)
                {
                    std::cerr<<"ERROR: cannot close attribute"<<std::endl;
                    std::exit(EXIT_FAILURE);
                }
                if (p->first != -1 && H5Sclose(p->first) < 0)
                {
                    std::cerr<<"ERROR: cannot close dataspace"<<std::endl;
                    std::exit(EXIT_FAILURE);
                }

                delete p;
                p = nullptr;
            });
    }

    managed_hid new_group()
    {
       return managed_hid(new std::pair<hid_t,hid_t>{-1,-1},
            [](std::pair<hid_t,hid_t>* p)
            {
                if (p == nullptr)
                    return;
                if (p->second != -1 && H5Gclose(p->second) < 0)
                {
                    std::cerr<<"ERROR: cannot close group"<<std::endl;
                    std::exit(EXIT_FAILURE);
                }
                if (p->first != -1 && H5Sclose(p->first) < 0)
                {
                    std::cerr<<"ERROR: cannot close dataspace"<<std::endl;
                    std::exit(EXIT_FAILURE);
                }

                delete p;
                p = nullptr;
            });

    }

    // convenience function to to unify handles to groups, attributes, datasets and the file itself
    hid_t get_dest(const managed_hid& dest) const
    {
        return dest ? dest->second : file;
    }

};

#endif //DG_HDF5FILE_H
