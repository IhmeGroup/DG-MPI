#include <Kokkos_Core.hpp>
#include <cmath>

using namespace Kokkos;
using team_policy = Kokkos::TeamPolicy<>;
using member_type = Kokkos::TeamPolicy<>::member_type;

KOKKOS_INLINE_FUNCTION
void set_to_0(){printf("0\n");}

KOKKOS_INLINE_FUNCTION
void set_to_1(){printf("1\n");}


class A {
public:

    // A(int k){
    //     set_pointer(k);
    // }

    // KOKKOS_INLINE_FUNCTION
    void set_pointer() const {


        // parallel_for(1, KOKKOS_CLASS_LAMBDA(const int& i){
        set = set_to_0;
        // });

        // set = set_to_0;
    }

    mutable void (*set)();
};

class TeamFunctor {
  public:
  TeamFunctor(){};

    void compute(A a){
        parallel_for("Compute1", TeamPolicy<>(10, Kokkos::AUTO), 
        KOKKOS_CLASS_LAMBDA(const TeamPolicy<>::member_type& member) {
            a.set_pointer();
            a.set();
      });
    }
};


int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    int N = argc > 1 ? atoi(argv[1]) : 1000000;
    int R = argc > 2 ? atoi(argv[2]) : 10;

    A a;
    a.set_pointer();
    TeamFunctor f;
    f.compute(a);
    Kokkos::fence();

  }
  Kokkos::finalize();
}
