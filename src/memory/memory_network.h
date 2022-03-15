#ifndef DG_MEMORY_NETWORK_H
#define DG_MEMORY_NETWORK_H

#include <mpi.h>
#include <Kokkos_Core.hpp>
#include "common/defines.h"
#include "mesh/mesh.h"

// Forward declaration
// TODO: Uncomment these for object-file build
//class Mesh;

/*
A network consisting of data and its methods on a distributed memory system.

This class is used to encapsulate raw MPI commands to abstract away low-level
message passing. Also, this class can help transfer data that may not be in
traditional C arrays (i.e. data in Kokkos views).  Some basic information about
the MPI status (the rank, the communicator...) lives here.
*/
class MemoryNetwork {
    public:
        // Constructor. Only pass in argc and argv when initializing MPI and
        // Kokkos is desired (can only be done once per MPI rank for a single
        // run of the program, so don't pass argc and argv twice). Otherwise, no
        // arguments should be supplied.
        MemoryNetwork(int argc = 0, char* argv[] = nullptr);
        void finalize();
        // Place an MPI barrier to synchronize ranks.
        void barrier() const;
        // Conduct an MPI gather all
        template<class T>
        void allgather(T send_data, T& recv_data);
        // Send the left and right states across partition boundaries.
        void communicate_face_solution(Kokkos::View<rtype***> UqL,
                Kokkos::View<rtype***> UqR, Kokkos::View<rtype***>* Uq_local,
                Kokkos::View<rtype***>* Uq_ghost, Mesh& mesh);
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
