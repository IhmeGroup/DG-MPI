#include <iostream>
#include "memory/memory_network.h"

using std::cout, std::endl;


MemoryNetwork::MemoryNetwork(int argc, char* argv[]) {
    // Initialize MPI
    MPI_Init(&argc, &argv);
    // Get the number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
    // Get the current rank
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // Print
    cout << "Rank " << rank << " / " << num_ranks << " reporting for duty!"
        << endl;
}

MemoryNetwork::~MemoryNetwork() {
    // Finalize MPI
    MPI_Finalize();
}
