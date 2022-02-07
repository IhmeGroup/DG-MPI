#include <iostream>
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
