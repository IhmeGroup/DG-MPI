#include "numerics/timestepping/stepper.h"
#include "solver/base.h"
#include "solver/tools.h"

template<unsigned dim>
void StepperBase<dim>::take_time_step(Solver<dim>& solver){
    throw InputException("Not Implemented Error");
}

template<unsigned dim>
void StepperBase<dim>::allocate_helpers(Solver<dim>& solver){
    throw InputException("Not Implemented Error");
}

template<unsigned dim>
void FE<dim>::allocate_helpers(Solver<dim>& solver){
    Kokkos::resize(dU, solver.res.extent(0),
        solver.res.extent(1), solver.res.extent(2));
}

template<unsigned dim>
void FE<dim>::take_time_step(Solver<dim>& solver){

    auto iMM_elems = solver.vol_helpers.iMM_elems;
    solver.get_residual();

    // Multiply by inverse mass matrix
    view_type_3D dU("dU", solver.res.extent(0), 
        solver.res.extent(1), solver.res.extent(2));
    SolverTools::mult_inv_mass_matrix(dt, iMM_elems, solver.res, dU);
   
    // solver.network.print_3d_view(iMM_elems);
    // solver.network.print_3d_view(dU);
    // printf("dt = %f\n", dt);
    // Copy back to host (TODO: Remove after debugging)
    // auto h_res = Kokkos::create_mirror_view_and_copy(
    //         Kokkos::DefaultHostExecutionSpace{}, solver.res);
    // auto h_dU = Kokkos::create_mirror_view_and_copy(
    //         Kokkos::DefaultHostExecutionSpace{}, dU);
    // solver.network.print_3d_view(h_res);
    // solver.network.print_3d_view(h_dU);

    //Update solution state
    const unsigned num_entries = dU.extent(0) * dU.extent(1) * dU.extent(2);
    Math::cApB_to_B(num_entries, 1., dU.data(), solver.Uc.data());
    solver.time += dt;

    // TODO: Add limiter call here? Or somewhere else?

}

template<unsigned dim>
void RK4<dim>::allocate_helpers(Solver<dim>& solver){
    Kokkos::resize(dU1, solver.res.extent(0),
        solver.res.extent(1), solver.res.extent(2));
    Kokkos::resize(dU2, solver.res.extent(0),
        solver.res.extent(1), solver.res.extent(2));
    Kokkos::resize(dU3, solver.res.extent(0),
        solver.res.extent(1), solver.res.extent(2));
    Kokkos::resize(dU4, solver.res.extent(0),
        solver.res.extent(1), solver.res.extent(2));
    Kokkos::resize(Uhold, solver.res.extent(0),
        solver.res.extent(1), solver.res.extent(2));
}

template<unsigned dim>
void RK4<dim>::take_time_step(Solver<dim>& solver){

    // unpack
    auto iMM_elems = solver.vol_helpers.iMM_elems;
    unsigned num_entries;

    Kokkos::deep_copy(Uhold, solver.Uc); // put the state coeffs in Utemp
    /* ---------------------------------------------- */
    /*                  First stage                   */
    /* ---------------------------------------------- */
    solver.get_residual();
    // Multiply by inverse mass matrix
    SolverTools::mult_inv_mass_matrix(dt, iMM_elems, solver.res, dU1);
    
    num_entries = dU1.extent(0) * dU1.extent(1) * dU1.extent(2);
    Math::cApB_to_C(num_entries, 0.5, dU1.data(), Uhold.data(), solver.Uc.data());
    // TODO: Add limiter call here? Or somewhere else?

    // printf("--------- STAGE 1 ---------\n");
    // Kokkos::deep_copy(solver.h_Uc, solver.Uc);
    // solver.network.print_3d_view(solver.h_Uc);
    // solver.network.print_3d_view(Uhold);

    /* ---------------------------------------------- */
    /*                 Second stage                   */
    /* ---------------------------------------------- */
    solver.time += dt / 2.0;
    solver.get_residual();
    SolverTools::mult_inv_mass_matrix(dt, iMM_elems, solver.res, dU2);

    num_entries = dU2.extent(0) * dU2.extent(1) * dU2.extent(2);
    Math::cApB_to_C(num_entries, 0.5, dU2.data(), Uhold.data(), solver.Uc.data());
    // TODO: Add limiter call here? Or somewhere else?

    // printf("--------- STAGE 2 ---------\n");
    // Kokkos::deep_copy(solver.h_Uc, solver.Uc);
    // solver.network.print_3d_view(solver.h_Uc);
    /* ---------------------------------------------- */
    /*                  Third stage                   */
    /* ---------------------------------------------- */
    solver.get_residual();
    SolverTools::mult_inv_mass_matrix(dt, iMM_elems, solver.res, dU3);

    num_entries = dU3.extent(0) * dU3.extent(1) * dU3.extent(2);
    Math::cApB_to_C(num_entries, 1.0, dU3.data(), Uhold.data(), solver.Uc.data());
    // TODO: Add limiter call here? Or somewhere else?    

    // printf("--------- STAGE 3 ---------\n");
    // Kokkos::deep_copy(solver.h_Uc, solver.Uc);
    // solver.network.print_3d_view(solver.h_Uc);
    /* ---------------------------------------------- */
    /*                 Fourth stage                   */
    /* ---------------------------------------------- */
    solver.time += dt / 2.0;
    solver.get_residual();
    SolverTools::mult_inv_mass_matrix(dt, iMM_elems, solver.res, dU4);

    num_entries = dU4.extent(0) * dU4.extent(1) * dU4.extent(2);
    Math::cApB_to_B(num_entries, 2.0, dU3.data(), dU4.data());
    Math::cApB_to_B(num_entries, 2.0, dU2.data(), dU4.data());
    Math::cApB_to_B(num_entries, 1.0, dU1.data(), dU4.data());
    Math::cApB_to_B(num_entries, 1.0/6.0, dU4.data(), Uhold.data());
    
    
    Kokkos::deep_copy(solver.Uc, Uhold);

    // printf("--------- STAGE 4 ---------\n");
    // Kokkos::deep_copy(solver.h_Uc, solver.Uc);
    // solver.network.print_3d_view(solver.h_Uc);
    // TODO: Add limiter call here? Or somewhere else?    
}

template class StepperBase<2>;
template class StepperBase<3>;
template class FE<2>;
template class FE<3>;
template class RK4<2>;
template class RK4<3>;

template class StepperFactory<2>;
template class StepperFactory<3>;
