#include <iostream>
#include <string>
#include <Kokkos_Core.hpp>
#include "common/defines.h"
#include "mesh/mesh.h"

using std::cout;
using std::endl;
using std::string;

inline MemoryNetwork::MemoryNetwork(int argc, char* argv[]) {
    // If arguments are provided, it is assumed that initializing MPI and Kokkos
    // should be done. Otherwise, it is skipped.
    if (argc != 0) {
        // Initialize MPI
        MPI_Init(&argc, &argv);
        // Initialize Kokkos (This needs to be after MPI_Init)
        Kokkos::initialize(argc, argv);
    }
    // Default communicator
    comm = MPI_COMM_WORLD;
    // Get the number of processes
    MPI_Comm_size(comm, (int*)&num_ranks);
    // Get the current rank
    MPI_Comm_rank(comm, (int*)&rank);
    if (rank == 0) { head_rank = true; }
    // Print
    cout << "Rank " << rank << " / " << num_ranks << " reporting for duty!"
        << endl;
}

inline void MemoryNetwork::finalize() {
    // Finalize Kokkos
    Kokkos::finalize();
    // Finalize MPI
    MPI_Finalize();
}

inline void MemoryNetwork::barrier() const {
    MPI_Barrier(comm);
}

inline void MemoryNetwork::communicate_face_solution(
        Kokkos::View<rtype***> UqL, Kokkos::View<rtype***> UqR, Mesh& mesh) {
    // Sizing
    auto nq = UqL.extent(1);
    auto ns = UqL.extent(2);

    // Arrays that need to be communicated
    double* ghost_Uq[mesh.num_neighbor_ranks];
    double* neighbor_Uq[mesh.num_neighbor_ranks];
    // Allocate
    for (unsigned neighbor_rank_idx = 0; neighbor_rank_idx <
            mesh.num_neighbor_ranks; neighbor_rank_idx++) {
        ghost_Uq[neighbor_rank_idx] = new double[
                mesh.num_faces_per_rank_boundary(neighbor_rank_idx) * nq * ns];
        neighbor_Uq[neighbor_rank_idx] = new double[
                mesh.num_faces_per_rank_boundary(neighbor_rank_idx) * nq * ns];
    }

    /* Copy face data into dense arrays */
    // Loop over neighboring ranks
    for (unsigned neighbor_rank_idx = 0; neighbor_rank_idx <
            mesh.num_neighbor_ranks; neighbor_rank_idx++) {
        // Loop over ghost faces on this rank boundary
        for (unsigned i = 0; i <
                mesh.num_faces_per_rank_boundary(neighbor_rank_idx); i++) {
            // Get local face ID of this ghost face
            // TODO
            //auto idx = mesh.global_to_local_iface_IDs.find(
            //        mesh.ghost_faces[neighbor_rank_idx][i]);
            //auto local_face_ID = mesh.global_to_local_iface_IDs.value_at(idx);
            unsigned local_face_ID = 0;
            // Is this rank on the left side?
            bool is_left = mesh.interior_faces(local_face_ID, 0)
                    == rank;

            // Copy data
            if (is_left) {
                for (unsigned j = 0; j < nq; j++) {
                    for (unsigned k = 0; k < ns; k++) {
                        ghost_Uq[neighbor_rank_idx][i*nq*ns + j*ns + k] =
                                UqL(local_face_ID, j, k);
                    }
                }
            } else {
                for (unsigned j = 0; j < nq; j++) {
                    for (unsigned k = 0; k < ns; k++) {
                        ghost_Uq[neighbor_rank_idx][i*nq*ns + j*ns + k] =
                                UqR(local_face_ID, j, k);
                    }
                }
            }
        }
    }

    /* Send data across ranks using MPI */
    // Loop over neighboring ranks
    for (unsigned neighbor_rank_idx = 0; neighbor_rank_idx <
            mesh.num_neighbor_ranks; neighbor_rank_idx++) {
        // Get rank of neighboring rank
        auto neighbor_rank = mesh.neighbor_ranks(neighbor_rank_idx);
        // Send ghost data
        MPI_Send(ghost_Uq[neighbor_rank_idx],
                mesh.num_faces_per_rank_boundary(neighbor_rank_idx) * nq * ns, MPI_RTYPE,
                neighbor_rank, rank, comm);
        // Receive neighbor data
        MPI_Recv(neighbor_Uq[neighbor_rank_idx],
                mesh.num_faces_per_rank_boundary(neighbor_rank_idx) * nq * ns, MPI_RTYPE,
                neighbor_rank, neighbor_rank, comm, MPI_STATUS_IGNORE);
    }

    /* Copy data from neighbors to local data structures */
    for (unsigned neighbor_rank_idx = 0; neighbor_rank_idx <
            mesh.num_neighbor_ranks; neighbor_rank_idx++) {
        // Loop over ghost faces on this rank boundary
        for (unsigned i = 0; i <
                mesh.num_faces_per_rank_boundary(neighbor_rank_idx); i++) {
            // Get local face ID of this ghost face
            // TODO
            //auto idx = mesh.global_to_local_iface_IDs.find(
            //        mesh.ghost_faces[neighbor_rank_idx][i]);
            //auto local_face_ID = mesh.global_to_local_iface_IDs.value_at(idx);
            unsigned local_face_ID = 0;
            // Is this rank on the left side?
            bool is_left = mesh.interior_faces(local_face_ID, 0)
                    == rank;
            // Copy data
            if (is_left) {
                for (unsigned j = 0; j < nq; j++) {
                    for (unsigned k = 0; k < ns; k++) {
                        UqR(local_face_ID, j, k) = neighbor_Uq[
                                neighbor_rank_idx][i*nq*ns + j*ns + k];
                    }
                }
            } else {
                for (unsigned j = 0; j < nq; j++) {
                    for (unsigned k = 0; k < ns; k++) {
                        UqL(local_face_ID, j, k) = neighbor_Uq[
                                neighbor_rank_idx][i*nq*ns + j*ns + k];
                    }
                }
            }
        }
    }

    // Free memory
    for (unsigned neighbor_rank_idx = 0; neighbor_rank_idx <
            mesh.num_neighbor_ranks; neighbor_rank_idx++) {
        delete [] ghost_Uq[neighbor_rank_idx];
        delete [] neighbor_Uq[neighbor_rank_idx];
    }
}

template <class T>
inline void MemoryNetwork::print(T data) const {
    barrier();
    if (head_rank) {
        cout << "--------------------------------- The Ranks Speak! --------------------------------" << endl;
    }
    // Loop over ranks
    for (unsigned i = 0; i < num_ranks; i++) {
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

template <class T>
inline void MemoryNetwork::print_view(Kokkos::View<T***> data) const {
    barrier();
    if (head_rank) {
        cout << "--------------------------------- The Ranks Speak! --------------------------------" << endl;
    }
    // Loop over ranks
    for (unsigned i = 0; i < num_ranks; i++) {
        barrier();
        // Only print on the appropriate rank
        if (i == rank) {
            // Print
            cout << "Rank " << rank << " says:" << endl;
            // Loop through indices of data
            for (long unsigned i = 0; i < data.extent(0); i++) {
                cout << "(" << i << ", :, :)" << endl;
                for (long unsigned j = 0; j < data.extent(1); j++) {
                    for (long unsigned k = 0; k < data.extent(2); k++) {
                        cout << data(i, j, k) << "  ";
                    }
                    cout << endl;
                }
                cout << endl;
            }
        }
    }
    barrier();
    if (head_rank) {
        cout << "------------------------------ The Ranks Have Spoken ------------------------------" << endl;
    }
}

template <class T>
inline void MemoryNetwork::print_view(Kokkos::View<T**> data) const {
    barrier();
    if (head_rank) {
        cout << "--------------------------------- The Ranks Speak! --------------------------------" << endl;
    }
    // Loop over ranks
    for (unsigned i = 0; i < num_ranks; i++) {
        barrier();
        // Only print on the appropriate rank
        if (i == rank) {
            // Print
            cout << "Rank " << rank << " says:" << endl;
            // Loop through indices of data
            for (long unsigned i = 0; i < data.extent(0); i++) {
                for (long unsigned j = 0; j < data.extent(1); j++) {
                    cout << data(i, j) << "  ";
                }
                cout << endl;
            }
            cout << endl;
        }
    }
    barrier();
    if (head_rank) {
        cout << "------------------------------ The Ranks Have Spoken ------------------------------" << endl;
    }
}

template <class T>
inline void MemoryNetwork::print_view(Kokkos::View<T*> data) const {
    barrier();
    if (head_rank) {
        cout << "--------------------------------- The Ranks Speak! --------------------------------" << endl;
    }
    // Loop over ranks
    for (unsigned i = 0; i < num_ranks; i++) {
        barrier();
        // Only print on the appropriate rank
        if (i == rank) {
            // Print
            cout << "Rank " << rank << " says:" << endl;
            // Loop through indices of data
            for (long unsigned i = 0; i < data.extent(0); i++) {
                cout << data(i) << "  ";
            }
            cout << endl;
        }
    }
    barrier();
    if (head_rank) {
        cout << "------------------------------ The Ranks Have Spoken ------------------------------" << endl;
    }
}
