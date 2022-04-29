#include "HDF5Wrapper.h"
#include <cstring>

int main(int argc, char **argv)
{

    int mpi_size, mpi_rank;
    MPI_Comm comm  = MPI_COMM_WORLD;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(comm, &mpi_size);
    MPI_Comm_rank(comm, &mpi_rank);

    {
        // Example 1:
        // - create a new file in parallel
        // - best practices for creating new attributes, groups and datasets
        // - efficient writing of parallel datasets

        // create and open a new file in parallel
        bool parallel = true;
        HDF5File file("sample_parallel.h5", H5F_ACC_TRUNC, parallel);

        // must be called by all ranks
        file.create_and_write_attribute("A1", 17);

        // write dataset to the file
        std::vector v{1, 2, 3};
        // must be called by all ranks. Name must be the same across all ranks, size and data are only taken from the head rank
        file.create_and_write_dataset("D1", v.size(), v.data());
        file.create_and_write_attribute("A_D1", 0.5);


        // must be called by all ranks
        auto g1 = file.create_group("g1");

        // write attribute to a group
        file.create_and_write_attribute("A2", 12.5, g1);

        // write dataset to a group
        std::vector v1{1.3, 1.6, 1.8};
        // must be called by all ranks. Name must be the same across all ranks, size and data are only taken from the head rank
        file.create_and_write_dataset("D2", v1.size(), v1.data(), g1);

        // write a multi-dimensional dataset
        int arr[2][3] = {{1,2,3}, {4,5,6}};
        // must be called by all ranks. Name must be the same across all ranks, dimensions and data are only taken from the head rank
        file.create_and_write_dataset("multi", {2,3}, &arr[0][0]);


        // next, each ranks writes its dataset into a large unified dataset. This is the most performant way to write large datasets
        std::vector dataset{1.,2.,3.};
        file.create_and_write_parallel_dataset("parallel_Dset", dataset.size(), dataset.data());

        // or alternatively, provide a flattened data buffer but also specify the dimensions of the original data to retrieve it later
        double multi_dataset[2][3] = {{1.,2.,3.}, {4.,5.,6.}};
        file.create_and_write_parallel_dataset("parallel_multi_Dset", {2,3}, &multi_dataset[0][0]);

    }

    auto print_vector = [](auto& v)
    {
        std::cout << "v: ";
        for(auto& e : v)
            std::cout << e << ' ';
        std::cout << '\n';
    };

    {
        // Example 2:
        // - open existing file in parallel
        // - best practice for reading the attributes and datasets created above
        // - reading datasets in parallel

        // open an existing file in parallel
        bool parallel = true;
        HDF5File file("sample_parallel.h5", H5F_ACC_RDWR, parallel);

        // retrieve attribute from file
        // for best parallel performance, use the version ending in '_all'
        int A1 = file.open_and_read_attribute_all<int>("A1");
        std::cout << "A1: " << A1 << '\n';

        // retrieve dataset from above. Two options:
        // 1) the data is only needed on one rank:
        if (mpi_rank == 0)
        {
            auto v = file.open_and_read_dataset<int>("D1");
            print_vector(v);
        }
        // 2) the data is needed on all ranks. Use the version ending in '_all'
        auto v = file.open_and_read_dataset_all<int>("D1");
        // now, each rank has a copy of 'v'

        // opening the group can also be done by a single rank, if te group will not be modified
        auto g1 = file.open_group("g1");

        if (mpi_rank == 0)
        {
            std::cout << "A2: " << file.open_and_read_attribute<double>("A2", g1) << '\n';
        }

        // retrieve multi-dimensional dataset. to read the same dataset to all ranks, use the '_all' version of the function
        auto [data, dims] = file.open_and_read_dataset_with_dims_all<int>("multi");
        int arr[2][3];
        // copy the flattened data in 'data' back into the 2D array 'arr'
        std::memcpy(&arr[0][0], data.data(), data.size()*sizeof(decltype(v)::value_type));
        if (mpi_rank == 0)
        {
            std::cout<<"dims =  {"<<dims[0]<<' '<<dims[1]<<"}\n";
            std::cout<<"data = { {"<<arr[0][0]<<' '<<arr[0][1]<<' '<<arr[0][2]<<"} {"<<arr[1][0]<<' '<<arr[1][1]<<' '<<arr[1][2]<<"} }\n";
        }


        // to retrieve the datasets written in parallel, some special functions are available:

        auto localSize = file.read_local_size_of_parallel_dataset("parallel_Dset"); // in this case returns 3
        auto totalSize = file.read_total_size_of_parallel_dataset("parallel_Dset"); // in this case returns mpi_size*3
        auto offset = file.read_offset_of_parallel_dataset("parallel_Dset"); // the offset of each rank in the dataset. Since each rank in this example has 3 elements, offset = 3*mpi_rank

        if (mpi_rank == 0)
        {
            std::cout<<"localSize = "<<localSize<<'\n';
            std::cout<<"totalSize = "<<totalSize<<'\n';
            std::cout<<"offset = "<<offset<<'\n';
        }

        // read the data block of the current process from the big dataset
        {
            // must be called by all ranks
            auto [data, dims] = file.read_dataset_parallel<double>("parallel_Dset");
            // data = {1,2,3}, dims = {3}; in this example, same for all ranks

            //or alternatively, pass a buffer to the function:
            std::vector<double> buffer(localSize);
            dims = file.read_dataset_parallel("parallel_Dset", buffer.data());
        }

        {
            // reading a multi-dimensional dataset has the same interface, but will return the flattened data
            auto [data, dims] = file.read_dataset_parallel<double>("parallel_multi_Dset");
            // data = {1,2,3,4,5,6}, dims = {2,3}
        }

    }
    std::cout<<"\n\n";
}
