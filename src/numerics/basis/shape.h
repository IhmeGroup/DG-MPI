#ifndef DG_NUMERICS_SHAPE_H
#define DG_NUMERICS_SHAPE_H

#include "common/defines.h"
#include "common/my_exceptions.h"
#include "common/enums.h"
#include "numerics/quadrature/tools.h"
#include "numerics/quadrature/segment.h"
#include "numerics/quadrature/quadrilateral.h"
#include "numerics/quadrature/hexahedron.h"

#include <Kokkos_Core.hpp>

namespace Basis {



int get_num_basis_coeff_segment(int p);

int get_num_basis_coeff_quadrilateral(int p);

void get_points_on_face_quadrilateral(const int face_id, const int orient, const int np,
        const Kokkos::View<rtype**>::HostMirror face_pts,
        Kokkos::View<rtype**>::HostMirror elem_pts);

int get_num_basis_coeff_hexahedron(int p);

class Shape {

public:

    /*
    Constructor
    */
    Shape(ShapeType shape_type);
    Shape() = default;
    ~Shape() = default;

    inline int get_NDIMS(){return NDIMS;}
    inline std::string get_name(){return name;}

    int (*get_num_basis_coeff)(int p);

    int get_quadrature_order(const int order);
    inline int get_num_faces_per_elem() const {return NFACES;}
    inline int get_num_orient_per_face() const {return NUM_ORIENT_PER_FACE;};
    void (*get_quadrature_data)(const int order, const int nq_1d,
        Kokkos::View<rtype**>::HostMirror& quad_pts,
        Kokkos::View<rtype*>::HostMirror& quad_wts);

    void (*get_points_on_face)(const int face_id, const int orient, const int np,
        const Kokkos::View<rtype**>::HostMirror face_pts,
        Kokkos::View<rtype**>::HostMirror elem_pts);

private:
    int (*get_quadrature_order_pointer)(const int order,
        const int NDIMS_);

protected:
    std::string name; // name of basis
    int NDIMS; // number of dimensions
    int NFACES; // number of faces
    int NCORNERS; // number of shape corners
    int NUM_ORIENT_PER_FACE; // number of orientations per face
    
};


} // end namespace Basis

#endif // DG_NUMERICS_SHAPE_H
