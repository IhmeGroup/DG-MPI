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
    virtual inline void set_time_step(rtype time_step) {dt = time_step;}
    virtual inline void set_final_time(rtype final_time) {tfinal = final_time;}
    virtual inline rtype get_time_step(){return dt;}
    virtual inline rtype get_tfinal(){return tfinal;}
    virtual void take_time_step(Solver<dim>& solver);


protected:
    rtype dt;
    rtype tfinal;

};


template<unsigned dim>
class FE : public StepperBase<dim> {
public:
    virtual inline void set_time_step(rtype time_step) override {dt = time_step;}
    virtual inline void set_final_time(rtype final_time) override {tfinal = final_time;}
    virtual inline rtype get_time_step() override {return dt;}
    virtual inline rtype get_tfinal() override {return tfinal;}
    virtual void take_time_step(Solver<dim>& solver) override;

protected:
    rtype dt;
    rtype tfinal;
};

template<unsigned dim>
class RK4 : public StepperBase<dim> {
public:
    virtual inline void set_time_step(rtype time_step) override {dt = time_step;}
    virtual inline void set_final_time(rtype final_time) override {tfinal = final_time;}
    virtual inline rtype get_time_step() override {return dt;}
    virtual inline rtype get_tfinal() override {return tfinal;}
    virtual void take_time_step(Solver<dim>& solver) override;

protected:
    rtype dt;
    rtype tfinal;
};

/*! \brief Factory for stepper object
 *
 */
template<unsigned dim>
class StepperFactory {
  public:
    static StepperBase<dim>* create_stepper(const StepperType type) {

        switch (type) {
            case StepperType::ForwardEuler:
                return new FE<dim>();
            case StepperType::RungeKutta4:
                return new RK4<dim>();
            default:
                throw InputException("StepperFactory could not create a stepper. "
                                     "The stepper type was not recognized");
        }
    }
};


#endif // end DG_NUMERICS_STEPPER_H