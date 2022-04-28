#ifndef DG_WRITER_H
#define DG_WRITER_H

#include <string>
#include <vector>
#include "mesh/mesh.h"
#include "memory/memory_network.h"
#include "solver/base.h"

class Writer {
    public:
        Writer(Mesh& mesh, MemoryNetwork& network, host_view_type_3D h_Uc,
            int nb, int ns, rtype time);
};

#endif //DG_WRITER_H
