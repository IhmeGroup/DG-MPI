#ifndef DG_WRITER_H
#define DG_WRITER_H

#include <string>
#include <vector>
#include "H5Cpp.h"
#include "mesh/mesh.h"

class Writer {
    public:
        Writer(const Mesh& mesh);

    private:
        template <class T>
        void write_dataset(T data, std::string name, H5::Group group,
                std::vector<hsize_t> dimensions);
        template <class T>
        void write_attribute(T data, std::string name, H5::H5File file);
        template <class T>
        H5::PredType get_hdf5_type();
};

#endif //DG_WRITER_H
