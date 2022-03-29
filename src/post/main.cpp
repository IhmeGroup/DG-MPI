#include <string>
#include <string>
#include <vector>
#include <memory>
#include <iostream>
#include "toml11/toml.hpp"
#include "mesh/mesh.h"
#include "utils/utils.h"
#include "io/writer.h"
#include "memory/memory_network.h"
#include "numerics/basis/basis.h"
#include "physics/base/base.h"
#include "numerics/numerics_data.h"
#include "io/io_params.h"
#include "solver/base.h"

using std::cout; using std::cin;
using std::endl; using std::string;

// Forward declaration
int main(int argc, char* argv[]);

template<unsigned dim>
void run_post(toml::value& toml_input, MemoryNetwork& network);

template<unsigned dim>
void get_error(Solver<dim>& solver, const int ord, bool normalize_by_volume);

template<unsigned dim>
void get_error(Solver<dim>& solver, const int ord, bool normalize_by_volume){

    // unpack
    auto time = solver.time;
    auto Uc = solver.Uc;
    auto mesh = solver.mesh;
    auto basis = solver.basis;
    auto physics = solver.physics;
    auto network = solver.network;
    auto vol_helpers = solver.vol_helpers;
    auto order = solver.order;

    rtype tot_vol_part;

    printf("order=%i\n", order);

    
    if (normalize_by_volume == true){
        tot_vol_part = MeshTools::get_total_volume(mesh.num_elems_part, 
            vol_helpers.vol_elems);
    } else {
        tot_vol_part = 1.;
    }

    printf("tot_vol_part=%f\n", tot_vol_part);
    view_type_2D error_elem("per element error", mesh.num_elems_part, physics.get_NS());

    // Over-integrate for the post processing -> compute quadrature
    int NDIMS = basis.shape.get_NDIMS();
    int nb = basis.get_num_basis_coeffs();
    int nq_1d; int nq;
    int qorder = basis.shape.get_quadrature_order(2*order);
    QuadratureTools::get_number_of_quadrature_points(qorder, NDIMS,
            nq_1d, nq);

    printf("nq(post)=%i\n", nq);
    view_type_2D quad_pts("quad_pts_post", nq, NDIMS);
    view_type_1D quad_wts("quad_wts_post", nq);
    host_view_type_2D h_quad_pts = Kokkos::create_mirror_view(quad_pts);
    host_view_type_1D h_quad_wts = Kokkos::create_mirror_view(quad_wts);

    basis.shape.get_quadrature_data(qorder, nq_1d, h_quad_pts, h_quad_wts);

    Kokkos::deep_copy(quad_pts, h_quad_pts);
    Kokkos::deep_copy(quad_wts, h_quad_wts);

    // Get the basis values evaluated at the 
    // overintegrated quadrature points
    view_type_2D basis_val("basis_val_post", nq, nb);
    view_type_2D gbasis_val("gbasis_val_post", nq, nb);
    host_view_type_2D h_basis_val = Kokkos::create_mirror_view(basis_val);
    host_view_type_2D h_gbasis_val = Kokkos::create_mirror_view(gbasis_val);

    basis.get_values(h_quad_pts, h_basis_val);
    mesh.gbasis.get_values(h_quad_pts, h_gbasis_val);

    Kokkos::deep_copy(basis_val, h_basis_val);
    Kokkos::deep_copy(gbasis_val, h_gbasis_val);

    // Get the geometric basis ref gradient evaluated
    // at the overintegrated quadrature points
    view_type_3D gbasis_ref_grad("gbasis_ref_grad_post", nq, nb, NDIMS);
    host_view_type_3D h_gbasis_ref_grad = Kokkos::create_mirror_view(gbasis_ref_grad);

    mesh.gbasis.get_grads(h_quad_pts, h_gbasis_ref_grad);
    Kokkos::deep_copy(gbasis_ref_grad, h_gbasis_ref_grad);

    // allocate Uq
    view_type_3D Uq("Uq", mesh.num_elems_part, 
        basis_val.extent(0), physics.get_NS());
    // get the solution evaluated at the quadrature points
    Kokkos::fence();
    // Evaluate the state
    VolumeHelpers::evaluate_state(mesh.num_elems_part,
        basis_val, Uc, Uq);
    Kokkos::fence();

    int scratch_size = scratch_view_2D_rtype::shmem_size(mesh.num_nodes_per_elem, mesh.dim)
        + scratch_view_2D_rtype::shmem_size(nq, NDIMS) 
        + scratch_view_2D_rtype::shmem_size(nq, physics.get_NS())
        + scratch_view_1D_rtype::shmem_size(nq)
        + scratch_view_3D_rtype::shmem_size(nq, NDIMS, NDIMS);

    Kokkos::parallel_for("init state", Kokkos::TeamPolicy<>( mesh.num_elems_part,
        Kokkos::AUTO).set_scratch_size( 1, Kokkos::PerTeam( scratch_size )),
        KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& member) {

            const int elem_ID = member.league_rank();

            // set the scratch memory for local memory
            scratch_view_2D_rtype elem_coords(member.team_scratch( 1 ),
                mesh.num_nodes_per_elem, NDIMS);

            scratch_view_2D_rtype xphys(member.team_scratch( 1 ),
                nq, NDIMS);

            scratch_view_2D_rtype u_exact(member.team_scratch( 1 ),
                nq, physics.NUM_STATE_VARS);

            scratch_view_1D_rtype djac(member.team_scratch( 1 ),
                nq);

            scratch_view_3D_rtype jac(member.team_scratch( 1 ),
                nq, NDIMS, NDIMS);

            // get the physical location of the evaluation points
            if (member.team_rank() == 0 ) {
                MeshTools::elem_coords_from_elem_ID(mesh, elem_ID, elem_coords,
                    member);
                MeshTools::ref_to_phys(gbasis_val, xphys, elem_coords, member);            
            }
            member.team_barrier();

            // Evaluate the initial condition at the evaluated coordinates in physical space
            Kokkos::parallel_for(Kokkos::TeamThreadRange(member, nq), [&] (const int iq) {
                physics.call_exact_solution(Kokkos::subview(xphys, iq, Kokkos::ALL()), time,
                    Kokkos::subview(u_exact, iq, Kokkos::ALL()));
            });
            member.team_barrier();

            // Evaluate djac
            BasisTools::get_element_jacobian(quad_pts, gbasis_ref_grad,
                jac, djac, elem_coords, member);

            member.team_barrier();

            // Get the error per element
            Kokkos::parallel_for(Kokkos::TeamThreadRange(member, physics.NUM_STATE_VARS), [&] (const int is) {
                rtype error_elem_is = 0.;
                Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(member, nq), [&] (const int iq, rtype& err_sum){
                    err_sum += pow((Uq(elem_ID, iq, is) - u_exact(iq, is)), ord) * quad_wts(iq) * djac(iq);
                    // printf("Uq(%i, %i, %i)=%f\n", elem_ID, iq, is, Uq(elem_ID, iq, is));
                    // printf("ue(%i, %i, %i)=%f\n", elem_ID, iq, is, u_exact(iq, is));
                }, error_elem_is);
                error_elem(elem_ID, is) = error_elem_is;
            });

        });


        network.barrier();

        rtype error_per_rank[physics.get_NS()] = {}; // init to zero
        for (unsigned is = 0; is < physics.get_NS(); is++){
            Kokkos::parallel_reduce("sum element errors", mesh.num_elems_part, 
                KOKKOS_LAMBDA ( const int ie, rtype &update ){
                    update += error_elem(ie, is);
            }, error_per_rank[is]);
        }

        // do an mpi gather here for both error_per_rank and tot_vol_rank
        rtype total_volume;
        rtype total_error[physics.get_NS()];
        network.allreduce(tot_vol_part, total_volume);
        network.barrier();

        for (unsigned is = 0; is < physics.get_NS(); is++){
            network.allreduce(error_per_rank[is], total_error[is]);
            network.barrier();
        
            total_error[is] = pow(total_error[is] / total_volume, 1./ord); 
            printf("norm_total_error(%i)=%.*e\n", is, DECIMAL_DIG, total_error[is]);
        }




}

template<unsigned dim>
void run_post(toml::value& toml_input, MemoryNetwork& network) {
    // TODO: Add gorder from mesh file
    int order = 1;
    // Get parameters related to the numerics
    auto numerics_params = Numerics::NumericsParams(toml_input, order);
    // Create mesh
    auto gbasis = Basis::Basis(numerics_params.gbasis, order);

    // TODO: Make a read_mesh only function that doesn't 
    //       recompute everything. i.e. we don't need all the 
    //       connectivity for this.
    auto mesh = Mesh(toml_input, network.num_ranks, network.rank,
            network.head_rank, gbasis);

    // Get physics type -> NOTE: physics object is constructed in solver constructor
    std::string phys = toml::find<std::string>(toml_input, "Physics", "name");
    auto physics_type = enum_from_string<PhysicsType>(phys.c_str());

    auto solver = Solver<dim>(toml_input, mesh, network, numerics_params,
        physics_type);

    // Read in InitialCondition data and copy it to the physics.IC_data view
    std::vector<rtype> IC_data_vec=toml::find<std::vector<rtype>>(toml_input, "InitialCondition", "data");
    assert(IC_data_vec.size()<=INIT_EX_PARAMS_MAX);

    Kokkos::resize(solver.physics.IC_data, IC_data_vec.size());
    host_view_type_1D h_IC_data = Kokkos::create_mirror_view(solver.physics.IC_data);
    // place initial condition data in host mirror view
    for (int i = 0; i<IC_data_vec.size(); i++){
        h_IC_data(i) = IC_data_vec[i];
    }
    // copy the initial condition data to the device
    Kokkos::deep_copy(solver.physics.IC_data, h_IC_data);

    // TODO: We don't need all the helpers for this. Limit to the needed ones.
    // Precompute Helpers
    solver.precompute_matrix_helpers();

    // Read in coefficients from data.h file
    std::string filename = "data.h5";
    solver.read_in_coefficients(filename);

    // Get error if desired
    // TODO: Add inputs to choose whether or not to calculate.
    // for now we will just always calculate (also, default to all state vars?)
    const int err_order = 2; // L2 error for now but leave flexibility for L1, etc...
    get_error(solver, err_order, true);


    // Finalize mesh
    mesh.finalize();

}

int main(int argc, char* argv[]) {

    MemoryNetwork network(argc, argv);

    string toml_fname = "input.toml";
    // If a different name is specified, use that
    if (Utils::exist_option(argv, argv + argc, "-input")) {
        toml_fname = string(Utils::get_option(argv, argv + argc, "-input"));
    }

    auto toml_input = toml::parse(toml_fname);

    // Run the requested post-processing
    const unsigned NDIMS = toml::find<unsigned>(toml_input, "Physics", "dim");
    if (NDIMS == 2){
        run_post<2>(toml_input, network);
    } else if (NDIMS == 3){
        run_post<3>(toml_input, network);
    }
    // Finalize memory network
    network.finalize();
}