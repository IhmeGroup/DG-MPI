#ifndef DG_MEMORY_NETWORK_H
#define DG_MEMORY_NETWORK_H

#include <mpi.h>

/*
A network consisting of data and its methods on a distributed memory system.

This class is used to encapsulate raw MPI commands to abstract away low-level
message passing. Also, this class can help transfer data that may not be in
traditional C arrays (i.e. data in Kokkos views).  Some basic information about
the MPI status (the rank, the communicator...) lives here.
*/
class MemoryNetwork {
    public:
        MemoryNetwork(int argc, char* argv[]);
        ~MemoryNetwork();

    public:
        int num_ranks;
        int rank;
        bool head_rank = false;
};

#endif //DG_MEMORY_NERWORK_H
