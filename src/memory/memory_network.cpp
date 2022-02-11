#include <iostream>
#include <string>
#include <Kokkos_Core.hpp>
#include "memory/memory_network.h"

using std::cout, std::endl;


MemoryNetwork::MemoryNetwork(int argc, char* argv[]) {
    // Initialize MPI
    MPI_Init(&argc, &argv);
    // Default communicator
    comm = MPI_COMM_WORLD;
    // Get the number of processes
    MPI_Comm_size(comm, &num_ranks);
    // Get the current rank
    MPI_Comm_rank(comm, &rank);
    if (rank == 0) { head_rank = true; }
    // Print
    cout << "Rank " << rank << " / " << num_ranks << " reporting for duty!"
        << endl;

    // Initialize Kokkos (This needs to be after MPI_Init)
    Kokkos::initialize(argc, argv);
}

MemoryNetwork::~MemoryNetwork() {
    // Finalize Kokkos
    Kokkos::finalize();
    // Finalize MPI
    MPI_Finalize();
}

void MemoryNetwork::barrier() const {
    MPI_Barrier(comm);
}

template <class T>
void MemoryNetwork::print(T data) const {
    barrier();
    if (head_rank) {
        cout << "--------------------------------- The Ranks Speak! --------------------------------" << endl;
    }
    // Loop over ranks
    for (int i = 0; i < num_ranks; i++) {
        barrier();
        // Only print on the appropriate rank
        if (i == rank) {
            // Print
            cout << "Rank " << rank << " says:" << endl;
            cout << data << endl;
        }
    }
    barrier();
    if (head_rank) {
        cout << "------------------------------ The Ranks Have Spoken ------------------------------" << endl;
    }
}

template void MemoryNetwork::print<int>(int) const;
template void MemoryNetwork::print<char*>(char*) const;
template void MemoryNetwork::print<const char*>(const char*) const;
template void MemoryNetwork::print<std::string>(std::string) const;
