#ifndef DG_MEMORY_NETWORK_H
#define DG_MEMORY_NETWORK_H

#include <mpi.h>
#include <Kokkos_Core.hpp>
#include "common/defines.h"

// Forward declaration
class Mesh;

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
        // Place an MPI barrier to synchronize ranks.
        void barrier() const;
        // Send the left and right states across partition boundaries.
        void communicate_face_solution(Kokkos::View<rtype***> UqL,
                Kokkos::View<rtype***> UqR, Mesh& mesh);
        template <class T>
        void print(T data) const;
        template <class T>
        void print_view(Kokkos::View<T***> data) const;
        template <class T>
        void print_view(Kokkos::View<T**> data) const;
        template <class T>
        void print_view(Kokkos::View<T*> data) const;

    public:
        unsigned num_ranks;
        unsigned rank;
        bool head_rank = false;
        MPI_Comm comm;
};

#include "memory/memory_network.cpp"

#endif //DG_MEMORY_NERWORK_H
