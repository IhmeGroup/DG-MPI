#ifndef DG_ENUMS_H
#define DG_ENUMS_H

#include <iostream>
#include <string>
#include <vector>
#include <sstream>

#include "common/my_exceptions.h"

using std::cout, std::endl;
using std::string, std::vector;

// This template is used to point towards the correct array containing the enum
// string names.
template <class T>
vector<string> get_enum_to_string_array();

// This template returns the string-valued name of an enum value. The template
// parameter should be set to the enum type.
// Example:
//     enum_to_string<BasisType>("Legendre2D")
template <class T>
string enum_to_string(T enum_value) {
    // Get array of enum to string
    vector<string> enum_to_string = get_enum_to_string_array<T>();
    // Return the string
    return enum_to_string[enum_value];
}

// This template returns the enum from its string-valued name. The template
// parameter should be set to the enum type.
// Example:
//     enum_from_string<BasisType>(BasisType::Legendre2D)
template <class T>
T enum_from_string(string str) {
    // Get array of enum to string
    vector<string> enum_to_string = get_enum_to_string_array<T>();
    // Find index of enum
    auto index_it = std::find(std::begin(enum_to_string),
            std::end(enum_to_string), str);
    // Check if the item exists in the array
    if (index_it == std::end(enum_to_string)) {
        // If it doesn't exist, then print the available choices and throw an
        // exception
        std::stringstream msg;
        msg << endl << endl << "Oh no! " << str << " is not a valid choice!" << endl;
        msg << "Instead, try one of these:" << endl;
        for (auto it = std::begin(enum_to_string); it != std::end(enum_to_string); it++) {
            msg << "    " << *it << endl;
        }
        throw InputException(msg.str());
    }
    // Get the index and return the proper enum
    auto index = std::distance(std::begin(enum_to_string), index_it);
    return T(index);
}

enum BasisType {
    LagrangeSeg,
    LagrangeQuad,
    LagrangeHex,
    LagrangeEq3D,
    LegendreSeg,
    LegendreQuad,
    LegendreHex,
    LagrangeGLL1D,
    LagrangeGLL2D,
    LagrangeGLL3D,
    LagrangeGL2D,
    LagrangeGL3D,
    TriangleLagrange
};
template <>
inline vector<string> get_enum_to_string_array<BasisType> () {
    return vector<string> {
        "LagrangeSeg",
        "LagrangeQuad",
        "LagrangeHex",
        "LegendreSeg",
        "LegendreQuad",
        "LegendreHex",
        "LagrangeEq3D",
        "LagrangeGLL1D",
        "LagrangeGLL2D",
        "LagrangeGLL3D",
        "LagrangeGL2D",
        "LagrangeGL3D",
        "TriangleLagrange"
    };
}

enum QuadratureType {
    GaussLegendre,
    GaussLobatto,
    TriangleDunavant,
    TriangleGaussLegendre
};
template <>
inline vector<string> get_enum_to_string_array<QuadratureType> () {
    return vector<string> {
        "GaussLegendre",
        "GaussLobatto",
        "TriangleDunavant",
        "TriangleGaussLegendre"
    };
}

enum NodeType {
    Equidistant, // equidistant nodes in reference space
    Interior // equidistant nodes but in the interior of the reference element, only useful for uniform quad/hex meshes
};
template <>
inline vector<string> get_enum_to_string_array<NodeType> () {
    return vector<string> {
        "Equidistant",
        "Interior"
    };
}

enum NormType {
    DomainIntegral,
    L1,
    L2
};
template <>
inline vector<string> get_enum_to_string_array<NormType> () {
    return vector<string> {
        "DomainIntegral",
        "L1",
        "L2"
    };
}

enum ShapeType {
    Segment,
    Quadrilateral,
    Hexahedron,
    Triangle
};
template <>
inline vector<string> get_enum_to_string_array<ShapeType> () {
    return vector<string> {
        "Segment",
        "Quadrilateral",
        "Hexahedron",
        "Triangle"
    };
}

enum DGSchemeType {
    DG,
    SumFact,
    LegendreStruct,
    ColocatedGL,
    ColocatedGLL,
    None
};
template <>
inline vector<string> get_enum_to_string_array<DGSchemeType> () {
    return vector<string> {
        "DG",
        "SumFact",
        "LegendreStruct",
        "ColocatedGL",
        "ColocatedGLL",
        "None"
    };
}

enum InitType {
    L2Projection,
    Interpolation
};
template <>
inline vector<string> get_enum_to_string_array<InitType> () {
    return vector<string> {
        "L2Projection",
        "Interpolation"
    };
}

#endif //DG_ENUMS_H
