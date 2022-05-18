#include "HDF5Wrapper.h"
#include <cstring>
int main()
{
    {
        // Example 1:
        // - create a new file
        // - create new attributes, groups and datasets

        // create and open a new file
        HDF5File file("sample_serial.h5", H5F_ACC_TRUNC);

        file.create_and_write_attribute("A1", 17);

        // write dataset to the file
        std::vector v{1, 2, 3};
        file.create_and_write_dataset("D1", v.size(), v.data());
        file.create_and_write_attribute("A_D1", 0.5);


        auto g1 = file.create_group("g1");

        // write attribute to a group
        file.create_and_write_attribute("A2", 12.5, g1);

        // write dataset to a group
        std::vector v1{1.3, 1.6, 1.8};
        file.create_and_write_dataset("D2", v1.size(), v1.data(), g1);

        // write a multi-dimensional dataset
        int arr[2][3] = {{1,2,3}, {4,5,6}};
        file.create_and_write_dataset("multi", {2,3}, &arr[0][0]);
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
        // - open existing file
        // - read the attributes and datasets created above

        // open an existing file
        HDF5File file("sample_serial.h5", H5F_ACC_RDWR);

        // retrieve attribute from file
        std::cout << "A1: " << file.open_and_read_attribute<int>("A1") << '\n';

        // retrieve dataset from above
        auto v = file.open_and_read_dataset<int>("D1");
        print_vector(v);

        std::cout << "A_D1: " << file.open_and_read_attribute<double>("A_D1") << '\n';

        auto g1 = file.open_group("g1");

        // read attribute from group
        std::cout << "A2: " << file.open_and_read_attribute<double>("A2", g1) << '\n';

        // retrieve dataset from group
        auto v1 = file.open_and_read_dataset<double>("D2", g1);
        print_vector(v1);

        // retrieve multi-dimensional dataset
        auto [data, dims] = file.open_and_read_dataset_with_dims<int>("multi");
        std::cout<<"dims =  {"<<dims[0]<<' '<<dims[1]<<"}\n";
        int arr[2][3];
        std::memcpy(&arr[0][0], data.data(), data.size()*sizeof(decltype(v)::value_type));
        std::cout<<"data = { {"<<arr[0][0]<<' '<<arr[0][1]<<' '<<arr[0][2]<<"} {"<<arr[1][0]<<' '<<arr[1][1]<<' '<<arr[1][2]<<"} }\n";

        // directly retrieve mutli-dimensional dataset into a provided buffer
        dims = file.open_and_read_dataset_with_dims<int>("multi", &arr[0][0]);
        std::cout<<"dims =  {"<<dims[0]<<' '<<dims[1]<<"}\n";
        std::cout<<"data = { {"<<arr[0][0]<<' '<<arr[0][1]<<' '<<arr[0][2]<<"} {"<<arr[1][0]<<' '<<arr[1][1]<<' '<<arr[1][2]<<"} }\n";

    }
    std::cout<<"\n\n";
}
