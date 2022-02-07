#include <iostream>
#include <Kokkos_Core.hpp>
#include "memory/memory_network.h"
#include "dumb.h"
using std::cout, std::endl;

// KOKKOS_FUNCTION
// void DumbConstruct::attempt_to_set(double* U){
//     printf("Attempt to set");
//     U[0] = 1.0;
//     U[1] = 2.0;
//     U[2] = 3.0;
//     U[3] = 4.0;
// }
using ExecSpace = Kokkos::CudaSpace::execution_space;
using range_policy = Kokkos::RangePolicy<ExecSpace>;

KOKKOS_INLINE_FUNCTION
void DumbConstruct::operator()(const int i) const{

    a(i, 0) = 1.0 * i;
    a(i, 1) = 1.0 * i * i;
    a(i, 2) = 1.0 * i * i * i;
    printf("HERE I AM");

}

DumbConstruct::DumbConstruct(int argc, char* argv[]) {

    // Initialize Kokkos (This needs to be after MPI_Init)
    Kokkos::initialize(argc, argv);

    Kokkos::View<double**> a("U", 10, 3);

    Kokkos::parallel_for(10, a);

    // double U[4];
    // U(0) = 1.1;
    cout<<"try"<<endl;

    // attempt_to_set(U);
    cout<<"Set view"<<endl;
}

DumbConstruct::~DumbConstruct(){};