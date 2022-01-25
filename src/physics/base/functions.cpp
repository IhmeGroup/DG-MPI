#include "common/defines.h"
#include "physics/base/data.h"
#include "physics/base/functions.h"

namespace BaseFcnType {


template<unsigned dim>
Uniform::Uniform(Physics::PhysicsBase<dim> &physics, const rtype *state){

    int NS = physics.get_NS();
    for (int is = 0; is < NS; is++){
        physics->state[is] = state[is];
    }
}

template<unsigned dim>
void get_state(Physics::PhysicsBase<dim> &physics, const rtype *x, const rtype *t){};


}