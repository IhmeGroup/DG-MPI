#include "solver/base.h"
#include "solver/helpers.h"
#include "solver/tools.h"
#include "numerics/basis/basis.h"
#include "numerics/basis/tools.h"

#include "common/defines.h"
#include <iostream>

// #include "physics/euler/functions.h"
// #include "physics/euler/data.h"


using namespace VolumeHelpers;

Solver::Solver(const toml::value &input_file, Mesh& mesh, MemoryNetwork& network,
    Numerics::NumericsParams& params, PhysicsType physics_type)
    : input_file{input_file}, mesh{mesh}, network{network}, params{params} {

    // initialize time to zero
    time = 0.0;
    const auto IC_name = toml::find<std::string>(input_file, "InitialCondition", "name");

    order = toml::find<int>(input_file, "Numerics", "order");
    basis = Basis::Basis(params.basis, order);
    physics = Physics::Physics(physics_type, mesh.dim, IC_name);
    // need to get physics.NUM_STATE_VARS set (compile time constant?)
    Kokkos::resize(Uc, mesh.num_elems_part, basis.get_num_basis_coeffs(), 4);
    Kokkos::resize(U_face, mesh.num_ifaces_part, basis.get_num_basis_coeffs(), 4);
}


void Solver::precompute_matrix_helpers() {

    // VolumeHelperFunctor functor;

    // need to get the sizes of things to pass into scratch memory
    int nb = basis.get_num_basis_coeffs();
    int nq_1d; int nq;
    int qorder = basis.shape.get_quadrature_order(2 * basis.get_order());
    QuadratureTools::get_number_of_quadrature_points(qorder, mesh.dim,
            nq_1d, nq);

    printf("nq=%i", nq);
    printf("nb=%i", nb);
    // set scratch memory size for iMM helper
    int scratch_size_iMM = scratch_view_2D_rtype::shmem_size(mesh.num_nodes_per_elem, mesh.dim)
        + scratch_view_2D_rtype::shmem_size(nb, nb) + scratch_view_2D_rtype::shmem_size(nq, nb);
    int scratch_size_vol = scratch_view_2D_rtype::shmem_size(mesh.num_nodes_per_elem, mesh.dim);

    printf("scratch_iMM=%i\n", scratch_size_iMM);
    printf("scratch_vol=%i\n", scratch_size_vol);
    vol_helpers.compute_inv_mass_matrices(scratch_size_iMM, mesh, basis);
    Kokkos::fence();


    vol_helpers.compute_volume_helpers(scratch_size_vol, mesh, basis);
    Kokkos::fence();

    host_view_type_3D h_xphys = Kokkos::create_mirror_view(vol_helpers.x_elems);
    host_view_type_3D h_iMM_elems = Kokkos::create_mirror_view(vol_helpers.iMM_elems);

    Kokkos::deep_copy(h_xphys, vol_helpers.x_elems);
    Kokkos::deep_copy(h_iMM_elems, vol_helpers.iMM_elems);

    for (int k = 0; k < h_xphys.extent(0); k++){
    for (int i = 0; i < h_xphys.extent(1); i++){
        for (int j=0; j< h_xphys.extent(2); j++){
            printf("xphys(%i, %i, %i)=%f\n", k, i, j, h_xphys(k, i, j));
        }
    }
}

    for (int k = 0; k < h_iMM_elems.extent(0); k++){
    for (int i = 0; i < h_iMM_elems.extent(1); i++){
        for (int j=0; j< h_iMM_elems.extent(2); j++){
            printf("iMM_elems(%i, %i, %i)=%f\n", k, i, j, h_iMM_elems(k, i, j));
        }
    }
}

}

void Solver::init_state_from_fcn(Mesh& mesh_local){

    // The current initialization assumes that we are using an L2 projection for the 
    // initial conditions.

    // we also pass in a local reference of mesh even though it is a part of the solver
    // object. I believe this has to do with mesh being defined by reference in solver
    // It may need to be stored by value in solver...
    
    // Over-integrate for the initial conditions -> compute quadrature
    int NDIMS = basis.shape.get_NDIMS();
    int nb = basis.get_num_basis_coeffs();
    int qorder = basis.shape.get_quadrature_order(2*order);
    int nq_1d; int nq;
    QuadratureTools::get_number_of_quadrature_points(qorder, NDIMS,
            nq_1d, nq);

    view_type_2D quad_pts("quad_pts", nq, NDIMS);
    view_type_1D quad_wts("quad_wts", nq);
    host_view_type_2D h_quad_pts = Kokkos::create_mirror_view(quad_pts);
    host_view_type_1D h_quad_wts = Kokkos::create_mirror_view(quad_wts);

    basis.shape.get_quadrature_data(qorder, nq_1d, h_quad_pts, h_quad_wts);

    Kokkos::deep_copy(quad_pts, h_quad_pts);
    Kokkos::deep_copy(quad_wts, h_quad_wts);

    // Get the basis values evaluated at the 
    // overintegrated quadrature points
    view_type_2D basis_val("basis_val", nq, nb);
    view_type_2D gbasis_val("gbasis_val", nq, nb);
    host_view_type_2D h_basis_val = Kokkos::create_mirror_view(basis_val);
    host_view_type_2D h_gbasis_val = Kokkos::create_mirror_view(gbasis_val);

    basis.get_values(h_quad_pts, h_basis_val);
    mesh.gbasis.get_values(h_quad_pts, h_gbasis_val);

    Kokkos::deep_copy(basis_val, h_basis_val);
    Kokkos::deep_copy(gbasis_val, h_gbasis_val);

    // Get the geometric basis ref gradient evaluated
    // at the overintegrated quadrature points
    view_type_3D gbasis_ref_grad("gbasis_ref_grad", nq, nb, NDIMS);
    host_view_type_3D h_gbasis_ref_grad = Kokkos::create_mirror_view(gbasis_ref_grad);

    mesh.gbasis.get_grads(h_quad_pts, h_gbasis_ref_grad);
    Kokkos::deep_copy(gbasis_ref_grad, h_gbasis_ref_grad);

    // set the scratch size for the parallel_for loop
    int scratch_size = scratch_view_2D_rtype::shmem_size(mesh.num_nodes_per_elem, mesh.dim)
        + scratch_view_2D_rtype::shmem_size(nq, NDIMS) 
        + scratch_view_2D_rtype::shmem_size(nq, physics.get_NS())
        + scratch_view_1D_rtype::shmem_size(nq)
        + scratch_view_3D_rtype::shmem_size(nq, NDIMS, NDIMS)
        + scratch_view_2D_rtype::shmem_size(nb, physics.get_NS());

    // NOTE: need to declare this prior to going into the parallel for because mesh 
    // needs to be changed so it can be passed by value ... I think
    const int num_nodes_per_elem = mesh.num_nodes_per_elem;

    auto physics_local = physics;

    Kokkos::parallel_for("init state", Kokkos::TeamPolicy<>( mesh.num_elems_part,
        Kokkos::AUTO).set_scratch_size( 1, Kokkos::PerTeam( scratch_size )),
        KOKKOS_CLASS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& member) {

            const int elem_ID = member.league_rank();

            // set the scratch memory for local memory
            scratch_view_2D_rtype elem_coords(member.team_scratch( 1 ),
                num_nodes_per_elem, NDIMS);

            scratch_view_2D_rtype xphys(member.team_scratch( 1 ),
                nq, NDIMS);

            scratch_view_2D_rtype f(member.team_scratch( 1 ),
                nq, physics.NUM_STATE_VARS);

            scratch_view_1D_rtype djac(member.team_scratch( 1 ),
                nq);

            scratch_view_3D_rtype jac(member.team_scratch( 1 ),
                nq, NDIMS, NDIMS);

            // get the physical location of the evaluation points
            if (member.team_rank() == 0 ) {
                MeshTools::elem_coords_from_elem_ID(mesh_local, elem_ID, elem_coords,
                    member);
                MeshTools::ref_to_phys(gbasis_val, xphys, elem_coords, member);            
            }
            member.team_barrier();

            // Evaluate the initial condition at the evaluated coordinates in physical space
            Kokkos::parallel_for(Kokkos::TeamThreadRange(member, nq), [&] (const int iq) {
                physics_local.call_IC(Kokkos::subview(xphys, iq, Kokkos::ALL()), time,
                    Kokkos::subview(f, iq, Kokkos::ALL()));
            });
            member.team_barrier();

            // Evaluate djac
            BasisTools::get_element_jacobian(quad_pts, gbasis_ref_grad,
                jac, djac, elem_coords, member);
            member.team_barrier();

            // L2 projection
            SolverTools::L2_projection(
                Kokkos::subview(vol_helpers.iMM_elems, elem_ID, Kokkos::ALL(), Kokkos::ALL()),
                basis_val, djac, quad_wts, f,
                Kokkos::subview(Uc, elem_ID, Kokkos::ALL(), Kokkos::ALL()), member);

        });




}
