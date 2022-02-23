#include <Kokkos_Core.hpp>
#include <cmath>

struct Functor {
  Kokkos::View<double*> a;
  Kokkos::View<double*>::HostMirror h_a;
  Functor() { foo(); }

  void foo() {
    Kokkos::resize(a,10);
    h_a = Kokkos::create_mirror_view(a);
    for(int i=0; i<10; i++) h_a(i) = i;
    Kokkos::deep_copy(a,h_a);
  }
  KOKKOS_FUNCTION
  void operator()(int i) const {

    // printf("%i %lf\n",i,a(i));
    int size = 4;
    int* tester = static_cast<int *>(Kokkos::kokkos_malloc<Kokkos::DefaultExecutionSpace>
      (size * sizeof(int)));
    // const int size = 4;
    // int tester[size];
    for (int i = 0 ; i < 4; i++){
      tester[i] = i;
    }

    Kokkos::View<int**, Kokkos::LayoutRight> vtester(tester, 2, 2);

    Kokkos::kokkos_free(tester);

    for (int i = 0; i < 2; i++){
      for (int j = 0; j < 2; j++){
          printf("vtester(%i, %i): %i\n", i, j, vtester(i, j));

      }
    }
  }
};
int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    int N = argc > 1 ? atoi(argv[1]) : 1000000;
    int R = argc > 2 ? atoi(argv[2]) : 10;
    Kokkos::parallel_for(1,Functor());
    Kokkos::fence();
  }
  Kokkos::finalize();
}