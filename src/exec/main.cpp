#include <string>
#include <iostream>
#include "toml11/toml.hpp"
#include "mesh/mesh.h"
#include "utils/utils.h"
#include "memory/memory_network.h"
#include "numerics/numerics_data.h"
#include "io/io_params.h"
#include "solver/base.h"

#include "memory/dumb.h"

using std::cout, std::endl, std::string;
using view_type = Kokkos::View<double * [3]>;

// parallel_for functor that fills the View given to its constructor.
// The View must already have been allocated.
struct InitView {
  view_type a;
  InitView(view_type a_) : a(a_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i) const {

    a(i, 0) = 1.0 * i;
    a(i, 1) = 1.0 * i * i;
    a(i, 2) = 1.0 * i * i * i;
  }
};

// Reduction functor that reads the View given to its constructor.
struct ReduceFunctor {
  view_type a;

  // Constructor takes View by "value"; this does a shallow copy.
  ReduceFunctor(view_type a_) : a(a_) {}

  // If you write a functor to do a reduction, you must specify the
  // type of the reduction result via a public 'value_type' alias.
  using value_type = double;

  KOKKOS_INLINE_FUNCTION
  void operator()(int i, double& lsum) const {
    lsum += a(i, 0) * a(i, 1) / (a(i, 2) + 0.1);
  }
};

int main(int argc, char* argv[]) {

    // DumbConstruct dumb(argc, argv);
    Kokkos::initialize(argc, argv);
    const int N = 10;

    view_type a("A", N);

    Kokkos::parallel_for(N, InitView(a));
    double sum = 0;
    Kokkos::parallel_reduce(N, ReduceFunctor(a), sum);
    printf("Result: %f\n", sum);





    // Initialize memory network
    MemoryNetwork network(argc, argv);

    // Default input file name
    string toml_fname = "input.toml";
    // If a different name is specified, use that
    if (Utils::exist_option(argv, argv + argc, "-input")) {
        toml_fname = string(Utils::get_option(argv, argv + argc, "-input"));
    }
    // this call is increasing build time by a lot!!
    // TODO think of something to reduce this (Kihiro 2021/03/04)
    auto toml_input = toml::parse(toml_fname);
    const int dim = toml::find<int>(toml_input, "Physics", "dim");
    //cout << toml_input.report() << endl;

    // TODO: This is just for testing
    auto numerics_params = Numerics::NumericsParams(toml_input, 3);
    auto solfile_params = SolutionFileParams(toml_input);

    // Create mesh
    auto mesh = Mesh(toml_input, network);

    // Create solver
    auto solver = Solver(toml_input, mesh, network);
}
