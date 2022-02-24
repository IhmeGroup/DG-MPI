#include <Kokkos_Core.hpp>
#include <cmath>

using team_policy=Kokkos::TeamPolicy<>;
using member_type=Kokkos::TeamPolicy<>::member_type;
using ScratchViewType=Kokkos::View< int**, Kokkos::DefaultExecutionSpace::scratch_memory_space,
      Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

using ScratchViewType2=Kokkos::View< double**, Kokkos::DefaultExecutionSpace::scratch_memory_space,
      Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

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
  void operator()(const member_type &teamMember) const {

    int size = 4;
    int size2 = 2;

    int size3 = 10;
    int size4 = 5;

    ScratchViewType vtester( teamMember.team_scratch( 0 ), size, size2);
    ScratchViewType2 vtester2( teamMember.team_scratch( 0 ), size3, size4);

    for (int i = 0; i < 4; i++){
      for (int j=0; j < 2; j++){
        vtester(i, j) = i;
      }
      // vtester[i] = i;
    }

    for (int i = 0; i < 4; i++){
      for (int j = 0; j < 2; j++){
      printf("vtester(%i): %i\n", i, vtester(i, j));
    }
  }

    for (int i = 0; i < 10; i++){
      for (int j=0; j < 5; j++){
        vtester2(i, j) = double(i)*2.;
      }
      // vtester[i] = i;
    }

    for (int i = 0; i < 10; i++){
      for (int j = 0; j < 5; j++){
      printf("vtester2(%i): %f\n", i, vtester2(i, j));
    }
  }
  }
};

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    int N = argc > 1 ? atoi(argv[1]) : 1000000;
    int R = argc > 2 ? atoi(argv[2]) : 10;

    int M1 = 4;
    int M2 = 2;
    int scratch_size = ScratchViewType::shmem_size( M1, M2 )
                      + ScratchViewType2::shmem_size( 10, 5);

    // for this sanity check we loop once and set out team size to one.
    Kokkos::parallel_for(team_policy( 1, 1 ).set_scratch_size( 0,
        Kokkos::PerThread( scratch_size ) ),
        Functor());


    Kokkos::fence();
  }
  Kokkos::finalize();
}
