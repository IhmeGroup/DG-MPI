#include <Kokkos_Core.hpp>

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    int N = argc > 1 ? atoi(argv[1]) : 1000000;
    int R = argc > 2 ? atoi(argv[2]) : 10;

    Kokkos::View<unsigned**> testing("testing", 8, 3);
    Kokkos::View<unsigned***> testing2("testing2", 8, 2, 5);

    printf("testing(%u, %u)\n", testing.extent(0), testing.extent(1));

    Kokkos::parallel_for(1, KOKKOS_LAMBDA ( int i) {

      printf("TEST 2D VIEW\n");

      printf("testing(%lu, %lu)\n", testing.extent(0), testing.extent(1));
      printf("testing(8, %u)\n", testing.extent(1));


      printf("TEST 3D VIEW\n");
      printf("testing(%lu, %lu, %lu)\n", testing.extent(0), testing.extent(1),
          testing.extent(2));
      printf("testing(%u, 2, 5)\n", testing.extent(0));
      printf("testing(8, %u, 5)\n", testing.extent(1));
      printf("testing(8, 2, %u)\n", testing.extent(2));


    });
  }
  Kokkos::finalize();
}
