#ifndef DG_WRITER_H
#define DG_WRITER_H

#include <string>
#include <vector>
#include "H5Cpp.h"
#include "mesh/mesh.h"
#include "memory/memory_network.h"
#include "solver/base.h"

class Writer {
    public:
        Writer(Mesh& mesh, MemoryNetwork& network, host_view_type_3D h_Uc,
            int nb, int ns);

    private:
        template <class T>
        void write_dataset(T data, std::string name, H5::Group group,
                std::vector<hsize_t> dimensions);
        template <class T>
        void write_attribute(T data, std::string name, H5::H5File file);

        template <class T>
        void write_attribute(T data, std::string name, H5::Group group);

        template <class T>
        H5::PredType get_hdf5_type();
};

#endif //DG_WRITER_H
