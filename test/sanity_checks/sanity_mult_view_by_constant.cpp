#include <Kokkos_Core.hpp>
#include <cmath>

#include <KokkosBatched_Scale_Decl.hpp>
#include <KokkosBatched_Scale_Impl.hpp>

using namespace Kokkos;
using namespace KokkosBatched;

using team_policy = Kokkos::TeamPolicy<>;
using member_type = Kokkos::TeamPolicy<>::member_type;


struct TeamFunctor {


    TeamFunctor(){

        Kokkos::resize(A, 2, 4);
        Kokkos::resize(b, 2);
        Kokkos::resize(c, 2);

        Kokkos::View<double**>::HostMirror h_A = Kokkos::create_mirror_view(A);
        Kokkos::View<double*>::HostMirror h_b = Kokkos::create_mirror_view(b);
        Kokkos::View<double*>::HostMirror h_c = Kokkos::create_mirror_view(c);

        int k = 0;
        for (int i = 0; i < 2; i++){
            for (int j = 0; j < 4; j++){
                h_A(i, j) = k;
                k++;

                printf("h_A(%i, %i)=%f\n", i, j, h_A(i, j));
            }
        }
        k = 0;
        for (int i = 0; i < 2; i++){ 
            h_b(i) = k;
            h_c(i) = k;
            k += 2;
        }

        Kokkos::deep_copy(A, h_A);
        Kokkos::deep_copy(b, h_b);
        Kokkos::deep_copy(c, h_c);
    };

    void compute(){
        parallel_for("Compute1", TeamPolicy<>(1, Kokkos::AUTO), *this);
    }

    KOKKOS_FUNCTION
    void operator()(const TeamPolicy<>::member_type& member) const {

        parallel_for(TeamThreadRange(member, 2), KOKKOS_LAMBDA (const int iq) {

            auto s_A = subview(A, iq, ALL());
            SerialScale::invoke(b(iq) * c(iq), s_A);
            // printf("iq:%i\n",iq);
            // TeamScale<member_type>::invoke(member, b(iq) * c(iq), s_A);
        });

        member.team_barrier();
    }


    Kokkos::View<double**> A;
    Kokkos::View<double*> b;
    Kokkos::View<double*> c;
};


int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        int N = argc > 1 ? atoi(argv[1]) : 1000000;
        int R = argc > 2 ? atoi(argv[2]) : 10;

        TeamFunctor f;
        f.compute();
        Kokkos::fence();
    
        View<double**>::HostMirror h_AA = create_mirror_view(f.A);
        deep_copy(h_AA, f.A);
        for (int i = 0; i < 2; i++){
            for (int j = 0; j < 4; j++){
                printf("h_AA(%i, %i)=%f\n", i, j, h_AA(i, j));
            }
        }   


    }
    Kokkos::finalize();
}