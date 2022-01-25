#include "physics/base/data.h"

template<unsigned dim>
void FcnBase::get_state(Physics::PhysicsBase<dim> &physics, const rtype *x, const rtype *t){
	throw NotImplementedException("FcnBase does not implement "
									"get_state -> implement in child class");
}