#include "numerics/timestepping/stepper.h"
#include "solver/base.h"

template<unsigned dim>
void StepperBase<dim>::take_time_step(Solver<dim>& solver){
    throw InputException("Not Implemented Error");
}

template<unsigned dim>
void FE<dim>::take_time_step(Solver<dim>& solver){


    solver.get_residual();

    // TODO: Multiply by inverse mass matrix
    // TODO: Update solution state

    // TODO: Add limiter call here? Or somewhere else?

}

template class StepperBase<2>;
template class StepperBase<3>;
template class FE<2>;
template class FE<3>;
