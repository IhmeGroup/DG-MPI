#ifndef DG_PHYSICS_BASE_FUNCTIONS_H
#define DG_PHYSICS_BASE_FUNCTIONS_H

#include "common/defines.h"
#include "physics/base/data.h"

namespace BaseFcnType {

enum class FcnType {
	Uniform,
	None,
};


class Uniform: public FcnBase {

public:
    template<unsigned dim>
    Uniform(Physics::PhysicsBase<dim> &physics, const rtype *state);
    ~Uniform();

    template<unsigned dim>
    void get_state(Physics::PhysicsBase<dim> &physics, const rtype *x, const rtype *t);
};

}
// enum class AnalyticType {
//     Uniform,
//     None,
// };


#endif //DG_PHYSICS_BASE_FUNCTIONS_H