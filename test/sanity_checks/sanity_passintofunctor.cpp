#include <Kokkos_Core.hpp>
#include <cmath>

using namespace Kokkos;
using team_policy = Kokkos::TeamPolicy<>;
using member_type = Kokkos::TeamPolicy<>::member_type;

class A {
public:
    int k;
    inline int get_k() const {return k;}
    inline void set_k(int i){k = i;}
};

class TeamFunctor {
  public:
  TeamFunctor(){};

    void compute(A a){
        parallel_for("Compute1", TeamPolicy<>(10, Kokkos::AUTO), 
        KOKKOS_CLASS_LAMBDA(const TeamPolicy<>::member_type& member) {
            printf("k=%i\n", a.get_k());
            printf("he");
      });
    }

};


int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    int N = argc > 1 ? atoi(argv[1]) : 1000000;
    int R = argc > 2 ? atoi(argv[2]) : 10;

    A a;
    a.set_k(5);

    TeamFunctor f;
    f.compute(a);
    Kokkos::fence();

    std::cout<<"poo"<<std::endl;
  }
  Kokkos::finalize();
}
