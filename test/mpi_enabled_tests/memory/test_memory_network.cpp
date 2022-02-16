#include <string>
#include <iostream>
#include "io/writer.h"
#include "memory/memory_network.h"

using std::string;
using Kokkos::ALL;

class MemoryTestSuite {
    public:
        // Name of test suite
        string test_suite_name = "MemoryTestSuite";
    private:
        // Command line arguments
        int argc;
        char** argv;

    public:
        // Constructor
        MemoryTestSuite(int argc, char* argv[]) : argc(argc), argv(argv) {
            // Run tests
            test_1();
        };

    void test_1() {
        string test_case_name = "ShouldSendFaceDataForFourQuads";

        // Create memory network
        auto network = MemoryNetwork(argc, argv);

        // Location of input file
        string toml_fname = string(PROJECT_ROOT) + "/test/mpi_enabled_tests/memory/input.toml";
        // Read input file
        auto toml_input = toml::parse(toml_fname);

        // Location of mesh file
        string mesh_file_name = string(PROJECT_ROOT) + "/test/mpi_enabled_tests/memory/quad_2x2.h5";
        // Create mesh
        auto mesh = Mesh(toml_input, network, mesh_file_name);

        // Sample left and right states
        constexpr int ns = 3;
        double UqL_i[ns];
        double UqR_i[ns];
        UqL_i[0] = 0; UqL_i[1] = 1; UqL_i[2] = 2;
        UqR_i[0] = 3; UqR_i[1] = 4; UqR_i[2] = 5;

        // Left and right states of all local interior faces
        int nq = 2;
        Kokkos::View<double***> UqL("UqL", mesh.num_ifaces_part, nq, ns);
        Kokkos::View<double***> UqR("UqR", mesh.num_ifaces_part, nq, ns);

        // Reminder: the interior faces are stored as shape
        // [num_ifaces_part, 8], where the 8 pieces of data are:
        // the rank, element ID, ref. face ID, and orientation on the left, and
        // the rank, element ID, ref. face ID, and orientation on the right.

        // Loop through faces, setting the face states to be these, if it's on
        // this rank
        Kokkos::parallel_for(mesh.num_ifaces_part, [&] KOKKOS_FUNCTION(const int& i) {
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

        // Perform communication across faces
        network.communicate_face_solution(UqL, UqR, mesh);

        network.print_view(UqL);
        network.print_view(UqR);
    };
};
