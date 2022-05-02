#include <iostream>
#include <string>
#include <Kokkos_Core.hpp>
#include "common/defines.h"
#include <iomanip>

using std::cout;
using std::endl;
using std::string;

inline MemoryNetwork::MemoryNetwork(int argc, char* argv[]) {
    // If arguments are provided, it is assumed that initializing MPI and Kokkos
    // should be done. Otherwise, it is skipped.
    bool should_initialize = argc != 0;

    if (should_initialize) {
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
    if (should_initialize) {
        cout << "Rank " << rank << " / " << num_ranks << " reporting for duty!"
            << endl;
    }
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

template<class T> 
inline void MemoryNetwork::allreduce(T send_data, T& recv_data) {
    MPI_Allreduce(&send_data, &recv_data, 1, MPI_RTYPE,
                MPI_SUM, MPI_COMM_WORLD);
}

inline void MemoryNetwork::communicate_face_solution(
        Kokkos::View<rtype***> UqL, Kokkos::View<rtype***> UqR,
        Kokkos::View<rtype***>* Uq_local_array,
        Kokkos::View<rtype***>* Uq_ghost_array, Mesh& mesh) {
    // Sizing
    auto nq = UqL.extent(1);
    auto ns = UqL.extent(2);

    /* Copy face data into dense arrays */
    // Loop over neighboring ranks
    for (unsigned neighbor_rank_idx = 0; neighbor_rank_idx <
            mesh.num_neighbor_ranks; neighbor_rank_idx++) {
        auto ghost_faces_view = mesh.ghost_faces[neighbor_rank_idx];
        auto Uq_local = Uq_local_array[neighbor_rank_idx];
        auto network_rank = rank;
        // Loop over ghost faces on this rank boundary
        Kokkos::parallel_for(
                mesh.h_num_faces_per_rank_boundary(neighbor_rank_idx),
                KOKKOS_LAMBDA(const unsigned& i) {
            // Get local face ID of this ghost face
            unsigned local_face_ID = mesh.get_local_iface_ID(
                    ghost_faces_view(i));

            // Is this rank on the left side?
            bool is_left = mesh.interior_faces(local_face_ID, 0) == network_rank;

            // Copy data
            // TODO: Maybe use deep_copy for this?
            if (is_left) {
                for (unsigned j = 0; j < nq; j++) {
                    for (unsigned k = 0; k < ns; k++) {
                        Uq_local(i, j, k) = UqL(local_face_ID, j, k);
                    }
                }
            } else {
                for (unsigned j = 0; j < nq; j++) {
                    for (unsigned k = 0; k < ns; k++) {
                        Uq_local(i, j, k) = UqR(local_face_ID, j, k);
                    }
                }
            }
        });
    }

    // TODO: Figure out nonblocking communication!!!
    // printf("after parallel_for loop for ghost face\n");
    /* Send data across ranks using MPI */
    // Loop over neighboring ranks
    for (unsigned neighbor_rank_idx = 0; neighbor_rank_idx <
            mesh.num_neighbor_ranks; neighbor_rank_idx++) {
        auto Uq_local = Uq_local_array[neighbor_rank_idx];
        auto Uq_ghost = Uq_ghost_array[neighbor_rank_idx];
        // Get rank of neighboring rank
        auto neighbor_rank = mesh.h_neighbor_ranks(neighbor_rank_idx);
        // TODO: Figure out CUDA-aware MPI. For now, just copy to the host and
        // back.
        auto send_view = Kokkos::create_mirror_view_and_copy(
                Kokkos::DefaultHostExecutionSpace{}, Uq_local);
        // auto recv_view = Kokkos::create_mirror_view_and_copy(
        //         Kokkos::DefaultHostExecutionSpace{}, Uq_ghost);
        // Send ghost data
        MPI_Request request;
        MPI_Isend(send_view.data(), Uq_local.size(),
                MPI_RTYPE, neighbor_rank, rank, comm, &request);
        // Receive neighbor data
        // MPI_Recv(recv_view.data(), Uq_ghost.size(),
        //         MPI_RTYPE, neighbor_rank, neighbor_rank, comm,
        //         MPI_STATUS_IGNORE);
        // Send back to device
        // Kokkos::deep_copy(Uq_ghost, recv_view);
    }

   // TODO: Figure out nonblocking communication!!!
    // printf("after parallel_for loop for ghost face\n");
    /* Send data across ranks using MPI */
    // Loop over neighboring ranks
    for (unsigned neighbor_rank_idx = 0; neighbor_rank_idx <
            mesh.num_neighbor_ranks; neighbor_rank_idx++) {
        auto Uq_local = Uq_local_array[neighbor_rank_idx];
        auto Uq_ghost = Uq_ghost_array[neighbor_rank_idx];
        // Get rank of neighboring rank
        auto neighbor_rank = mesh.h_neighbor_ranks(neighbor_rank_idx);
        // TODO: Figure out CUDA-aware MPI. For now, just copy to the host and
        // back.
        // auto send_view = Kokkos::create_mirror_view_and_copy(
        //         Kokkos::DefaultHostExecutionSpace{}, Uq_local);
        auto recv_view = Kokkos::create_mirror_view_and_copy(
                Kokkos::DefaultHostExecutionSpace{}, Uq_ghost);
        // Send ghost data
        // MPI_Send(send_view.data(), Uq_local.size(),
        //         MPI_RTYPE, neighbor_rank, rank, comm);
        // Receive neighbor data
        MPI_Recv(recv_view.data(), Uq_ghost.size(),
                MPI_RTYPE, neighbor_rank, neighbor_rank, comm,
                MPI_STATUS_IGNORE);
        // Send back to device
        Kokkos::deep_copy(Uq_ghost, recv_view);
    }

    barrier();

    /* Copy data from neighbors to local data structures */
    for (unsigned neighbor_rank_idx = 0; neighbor_rank_idx <
            mesh.num_neighbor_ranks; neighbor_rank_idx++) {
        auto ghost_faces_view = mesh.ghost_faces[neighbor_rank_idx];
        auto Uq_local = Uq_local_array[neighbor_rank_idx];
        auto Uq_ghost = Uq_ghost_array[neighbor_rank_idx];
        auto network_rank = rank;
        // Loop over ghost faces on this rank boundary
        Kokkos::parallel_for(
                mesh.h_num_faces_per_rank_boundary(neighbor_rank_idx),
                KOKKOS_LAMBDA(const unsigned& i) {
            // Get local face ID of this ghost face
            unsigned local_face_ID = mesh.get_local_iface_ID(
                    ghost_faces_view(i));

            // Is this rank on the left side?
            bool is_left = mesh.interior_faces(local_face_ID, 0) == network_rank;
            // Copy data
            if (is_left) {
                for (unsigned j = 0; j < nq; j++) {
                    for (unsigned k = 0; k < ns; k++) {
                        UqR(local_face_ID, j, k) = Uq_ghost(i, j, k);
                    }
                }
            } else {
                for (unsigned j = 0; j < nq; j++) {
                    for (unsigned k = 0; k < ns; k++) {
                        UqL(local_face_ID, j, k) = Uq_ghost(i, j, k);
                    }
                }
            }
        });
    }
    // printf("after copy\n");
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
inline void MemoryNetwork::print_view(Kokkos::View<T****> data) const {
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
                cout << "(" << i << ", :, :, :)" << endl;
                for (long unsigned j = 0; j < data.extent(1); j++) {
                    for (long unsigned k = 0; k < data.extent(2); k++) {
                        for (long unsigned l = 0; l < data.extent(3); l++){
                            cout << data(i, j, k, l) << "  ";
                        }
                        cout << endl;
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
inline void MemoryNetwork::print_view(Kokkos::View<T***> data) const {
    barrier();
    if (head_rank) {
        cout << "--------------------------------- The Ranks Speak! --------------------------------" << endl;
        cout << "View Shape ["<<data.extent(0)<<", "<<data.extent(1)<<", "<<
            data.extent(2) << "]"<<endl;
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
inline void MemoryNetwork::print_view(Kokkos::View<T***,
        Kokkos::DefaultExecutionSpace::scratch_memory_space,
        Kokkos::MemoryTraits<Kokkos::Unmanaged>> data) const {
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
inline void MemoryNetwork::print_view(Kokkos::View<T**,
        Kokkos::DefaultExecutionSpace::scratch_memory_space,
        Kokkos::MemoryTraits<Kokkos::Unmanaged>> data) const {
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


template <class ViewType>
inline void MemoryNetwork::print_3d_view(ViewType data) const {
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
                        cout << fixed << std::setprecision(12) << data(i, j, k) << "  ";
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

template <class ViewType>
inline void MemoryNetwork::print_4d_view(ViewType data) const {
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
                        for (long unsigned l = 0; l < data.extent(3); l++){
                            cout << fixed << std::setprecision(12) << data(i, j, k, l) << "  ";
                        }
                        cout << endl;
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