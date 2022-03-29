#ifndef DG_NUMERICS_STEPPER_H
#define DG_NUMERICS_STEPPER_H

#include "common/defines.h"
#include "common/my_exceptions.h"
#include "common/enums.h"

#include <Kokkos_Core.hpp>

// Forward declaration of solver
template<unsigned dim>
class Solver;

template<unsigned dim>
class StepperBase {

public:
    virtual void take_time_step(Solver<dim>& solver);


protected:
    rtype dt;

};


template<unsigned dim>
class FE : public StepperBase<dim> {
public:
    virtual void take_time_step(Solver<dim>& solver) override;
};

#endif // end DG_NUMERICS_STEPPER_H