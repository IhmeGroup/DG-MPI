#include <string>
#include <iostream>
#include "io/writer.h"
#include "memory/memory_network.h"
#include "mesh/mesh.h"
#include "../test/mpi_enabled_tests/memory/test_memory_network.h"

using Kokkos::ALL;
using std::cout;
using std::endl;
using std::string;

MemoryTestSuite::MemoryTestSuite() {
    // Run tests
    test_1();
}

void MemoryTestSuite::test_1() {
    string test_case_name = "ShouldSendFaceDataForFourQuads";
    double atol = 1e-8;

    // Create memory network
    auto network = MemoryNetwork();

    // Location of input file
    string toml_fname = string(PROJECT_ROOT) + "/test/mpi_enabled_tests/memory/input.toml";
    // Read input file
    auto toml_input = toml::parse(toml_fname);

    // Location of mesh file
    string mesh_file_name = string(PROJECT_ROOT) + "/test/mpi_enabled_tests/memory/quad_2x2.h5";
    // string mesh_file_name = string(PROJECT_ROOT) + "/examples/quad_2x2/some_big_mesh.h5";
    // Create mesh
    auto gbasis = Basis::Basis();
    auto mesh = Mesh(toml_input, network.num_ranks, network.rank,
            network.head_rank, gbasis, mesh_file_name);

    // Sample left and right states
    constexpr int ns = 4; //3;
    double UqL_i[ns];
    double UqR_i[ns];
    // UqL_i[0] = 0; UqL_i[1] = 1; UqL_i[2] = 2;
    // UqR_i[0] = 3; UqR_i[1] = 4; UqR_i[2] = 5;

    UqL_i[0] = 1.0; UqL_i[1] = 2.0; UqL_i[2] = 3.0; UqL_i[3] = 4.0;
    UqR_i[0] = 1.0; UqR_i[1] = 2.0; UqR_i[2] = 3.0; UqR_i[3] = 4.0;

    // Left and right states of all local interior faces
    int nq = 2;
    Kokkos::View<rtype***> UqL("UqL", mesh.num_ifaces_part, nq, ns);
    Kokkos::View<rtype***> UqR("UqR", mesh.num_ifaces_part, nq, ns);
    // Face local and ghost states
    auto Uq_local = new Kokkos::View<rtype***>[mesh.num_neighbor_ranks];
    auto Uq_ghost = new Kokkos::View<rtype***>[mesh.num_neighbor_ranks];
    for (unsigned i = 0; i < mesh.num_neighbor_ranks; i++) {
        Kokkos::resize(Uq_local[i], mesh.h_num_faces_per_rank_boundary(i), nq, ns);
        Kokkos::resize(Uq_ghost[i], mesh.h_num_faces_per_rank_boundary(i), nq, ns);
    }

    // Reminder: the interior faces are stored as shape
    // [num_ifaces_part, 8], where the 8 pieces of data are:
    // the rank, element ID, ref. face ID, and orientation on the left, and
    // the rank, element ID, ref. face ID, and orientation on the right.

    // Loop through faces, setting the face states to be these, if it's on
    // this rank
    Kokkos::parallel_for(mesh.num_ifaces_part, KOKKOS_LAMBDA(const int& i) {
            // Loop over left first, then right
            for (int rank_idx : {0, 4}) {
                if (mesh.interior_faces(i, rank_idx) == network.rank) {
                    // Loop over quadrature points
                    for (int j = 0; j < nq; j++) {
                        // Loop over state variables
                        for (int k = 0; k < ns; k++) {
                            // Set the state
                            if (rank_idx == 0) {
                                UqL(i, j, k) = UqL_i[k];
                            } else {
                                UqR(i, j, k) = UqR_i[k];
                            }
                        }
                    }
                }
            }
    });

    // std::cout<<"UqL"<<std::endl;
    // network.print_view(UqL);
    // std::cout<<"UqR"<<std::endl;
    // network.print_view(UqR);

    // Perform communication across faces
    network.communicate_face_solution(UqL, UqR, Uq_local, Uq_ghost, mesh);


    // std::cout<<"UqL"<<std::endl;
    // network.print_view(UqL);
    // std::cout<<"UqR"<<std::endl;
    // network.print_view(UqR);

    // Copy back to host
    auto h_UqL = Kokkos::create_mirror_view_and_copy(
            Kokkos::DefaultHostExecutionSpace{}, UqL);
    auto h_UqR = Kokkos::create_mirror_view_and_copy(
            Kokkos::DefaultHostExecutionSpace{}, UqR);

    // Check the answer
    bool passed_test = true;
    for (int i = 0; i < mesh.num_ifaces_part; i++) {
        for (int j = 0; j < nq; j++) {
            for (int k = 0; k < ns; k++) {
                if (h_UqL(i, j, k) - UqL_i[k] > atol) {
                    cout << "h_UqL(" << i << ", " << j << ", " << k << ") is wrong!" << endl;
                    passed_test = false;
                }
                if (h_UqR(i, j, k) - UqR_i[k] > atol) {
                    cout << "h_UqR(" << i << ", " << j << ", " << k << ") is wrong!" << endl;
                    passed_test = false;
                }
            }
        }
    }

    // Print final result
    string result;
    if (passed_test) { result = "passed"; }
    else { result = "failed"; }
    cout << "Test case \"" << test_case_name << "\" " << result << "! (rank "
            << network.rank << "/" << network.num_ranks << ")" << endl;

    // Cleanup
    Kokkos::fence();
    for (unsigned i = 0; i < mesh.num_neighbor_ranks; i++) {
        // Explicitly destruct inner views to avoid memory leak
        using Kokkos::View;
        Uq_local[i].~View<rtype***>();
        Uq_ghost[i].~View<rtype***>();
    }
    mesh.finalize();
}
