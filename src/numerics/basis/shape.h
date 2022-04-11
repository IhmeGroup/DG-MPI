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


/* --------------------------------------
        Segment Shape Definitions
----------------------------------------*/
inline
int get_num_basis_coeff_segment(int p);

inline
unsigned get_num_nodes_per_face_segment(const unsigned gorder);

inline
unsigned get_num_nodes_per_elem_segment(const unsigned gorder);


/* --------------------------------------
        Quadrilateral Shape Definitions
----------------------------------------
 *
 *         f2
 *      n3----n2
 *  f3  |      | f1
 *      n0----n1
 *         f0
 */
inline
int get_num_basis_coeff_quadrilateral(int p);

inline
unsigned get_num_nodes_per_face_quadrilateral(const unsigned gorder);

inline
unsigned get_num_nodes_per_elem_quadrilateral(const unsigned gorder);

inline
void get_points_on_face_quadrilateral(const int face_id, const int orient, const int np,
        const Kokkos::View<rtype**>::HostMirror face_pts,
        Kokkos::View<rtype**, Kokkos::LayoutStride>::HostMirror elem_pts);

KOKKOS_INLINE_FUNCTION
void get_face_pts_order_wrt_orient0_quadrilateral(const int orient, const int npts,
        Kokkos::View<int*> pts_order);

KOKKOS_INLINE_FUNCTION
void get_local_nodes_on_face_quadrilateral(const int face_id, const int gorder,
    scratch_view_1D_int lfnodes);

template<typename ViewType1D, typename ViewType2D> KOKKOS_INLINE_FUNCTION
void get_normals_on_face_quadrilateral(const int orient, const int np, const int gorder,
        const ViewType2D face_pts, const ViewType1D coeffs, ViewType1D normals);

/* --------------------------------------
        Hexahedron Shape Definitions
-----------------------------------------
 *         n2----n3
 *        / |    /|
 *       / n0---/n1
 *      n6----n7 /
 *      |/     |/
 *      n4----n5
 *
 * Faces:
 *  - 0 : 0, 2, 3, 1
 *  - 1 : 0, 1, 5, 4
 *  - 2 : 1, 3, 7, 5
 *  - 3 : 3, 2, 6, 7
 *  - 4 : 2, 0, 4, 6
 *  - 5 : 4, 5, 7, 6
 *
 *  Face Orientations:
 *
 *      n2----n3    n0----n2    n1----n0    n3----n1
 *      |   0  |    |   1  |    |   2  |    |   3  |
 *      n0----n1    n1----n3    n3----n2    n2----n0
 *
 *      n1----n3    n3----n2    n2----n0    n0----n1
 *      |   4  |    |   5  |    |   6  |    |   7  |
 *      n0----n2    n1----n0    n3----n1    n2----n3
 *
 */
inline
int get_num_basis_coeff_hexahedron(int p);

inline
unsigned get_num_nodes_per_face_hexahedron(const unsigned gorder);

inline
unsigned get_num_nodes_per_elem_hexahedron(const unsigned gorder);

inline
void get_points_on_face_hexahedron(const int face_id, const int orient, const int np,
        const Kokkos::View<rtype**>::HostMirror face_pts,
        Kokkos::View<rtype**, Kokkos::LayoutStride>::HostMirror elem_pts);

KOKKOS_INLINE_FUNCTION
void get_face_pts_order_wrt_orient0_hexahedron(const int orient, const int npts,
        Kokkos::View<int*> pts_order);

KOKKOS_INLINE_FUNCTION
void get_local_nodes_on_face_hexahedron(const int face_id, const int gorder,
    scratch_view_1D_int lfnodes);

template<typename ViewType1D, typename ViewType2D> KOKKOS_INLINE_FUNCTION
void get_normals_on_face_hexahedron(const int orient, const int np, const int gorder,
        const ViewType2D face_pts, const ViewType1D coeffs, ViewType1D normals);

class Shape {

public:

    /*
    Constructor
    */
    inline
    Shape(ShapeType shape_type);
    Shape() = default;
    ~Shape() = default;

    inline int get_NDIMS(){return NDIMS;}
    inline std::string get_name(){return name;}

    int (*get_num_basis_coeff)(int p);

    inline
    int get_quadrature_order(const int order);
    inline int get_num_faces_per_elem() const {return NFACES;}
    inline int get_num_orient_per_face() const {return NUM_ORIENT_PER_FACE;};
    unsigned (*get_num_nodes_per_face)(const unsigned gorder);
    unsigned (*get_num_nodes_per_elem)(const unsigned gorder);

    void (*get_quadrature_data)(const int order, const int nq_1d,
        Kokkos::View<rtype**>::HostMirror& quad_pts,
        Kokkos::View<rtype*>::HostMirror& quad_wts);

    void (*get_points_on_face)(const int face_id, const int orient, const int np,
        const Kokkos::View<rtype**>::HostMirror face_pts,
        Kokkos::View<rtype**, Kokkos::LayoutStride>::HostMirror elem_pts);

    KOKKOS_INLINE_FUNCTION
    void get_face_pts_order_wrt_orient0(const int orient, const int npts,
        Kokkos::View<int*> pts_order) const;

    KOKKOS_INLINE_FUNCTION
    void get_local_nodes_on_face(const int face_id, const int gorder,
        scratch_view_1D_int lfnodes) const;

    template<typename ViewType1D, typename ViewType2D> KOKKOS_INLINE_FUNCTION
    void get_normals_on_face(const int orient, const int np, const int gorder,
        const ViewType2D face_pts,
        const ViewType1D coeffs, ViewType1D normals) const;

private:
    int (*get_quadrature_order_pointer)(const int order,
        const int NDIMS_);

protected:
    ShapeType type; // name of shape type
    std::string name; // name of shape
    int NDIMS; // number of dimensions
    int NFACES; // number of faces
    int NCORNERS; // number of shape corners
    int NUM_ORIENT_PER_FACE; // number of orientations per face
    
};


} // end namespace Basis

#include "numerics/basis/shape.cpp"

#endif // DG_NUMERICS_SHAPE_H
