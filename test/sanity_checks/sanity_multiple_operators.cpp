#include <Kokkos_Core.hpp>
#include <cmath>

using namespace Kokkos;
using team_policy = Kokkos::TeamPolicy<>;
using member_type = Kokkos::TeamPolicy<>::member_type;

struct Functor {
  class Tag1{};
  class Tag2{};

  Functor(){};

  void compute(){
    parallel_for("Compute1", RangePolicy<Tag1>(0, 10), *this);
    parallel_for("Compute2", RangePolicy<Tag2>(0, 2), *this);
  }
  KOKKOS_FUNCTION
  void operator()(Tag1, int i) const {
    printf("Tag1\n");
  }

  KOKKOS_FUNCTION
  void operator()(Tag2, int i) const {
    printf("Tag2\n");
  }
};

struct TeamFunctor {
  class Tag1{};
  class Tag2{};

  TeamFunctor(){};

  void compute(){
    parallel_for("Compute1", TeamPolicy<Tag1>(10, Kokkos::AUTO), *this);
    parallel_for("Compute2", TeamPolicy<Tag2>(2, Kokkos::AUTO), *this);
  }
  KOKKOS_FUNCTION
  void operator()(Tag1, const TeamPolicy<>::member_type& member) const {
    printf("Tag1\n");
  }

  KOKKOS_FUNCTION
  void operator()(Tag2, const TeamPolicy<>::member_type& member) const {
    printf("Tag2\n");
  }
};


int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    int N = argc > 1 ? atoi(argv[1]) : 1000000;
    int R = argc > 2 ? atoi(argv[2]) : 10;

    Functor functor;
    functor.compute();
    Kokkos::fence();

    TeamFunctor functor2;
    functor2.compute();
    Kokkos::fence();
  }
  Kokkos::finalize();
}
