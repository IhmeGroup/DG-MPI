#ifndef DG_ENUMS_H
#define DG_ENUMS_H

#include <iostream>
#include <string>
#include <vector>
#include <sstream>

#include "common/my_exceptions.h"

// This template is used to point towards the correct array containing the enum
// string names.
template <class T>
std::vector<std::string> get_enum_to_string_array();

// This template returns the string-valued name of an enum value. The template
// parameter should be set to the enum type.
// Example:
//     enum_to_string<BasisType>("Legendre2D")
template <class T>
std::string enum_to_string(T enum_value) {
    // Get array of enum to string
    std::vector<std::string> enum_to_string = get_enum_to_string_array<T>();
    // Return the string
    return enum_to_string[enum_value];
}

// This template returns the enum from its string-valued name. The template
// parameter should be set to the enum type.
// Example:
//     enum_from_string<BasisType>(BasisType::Legendre2D)
template <class T>
T enum_from_string(std::string str) {
    // Get array of enum to string
    std::vector<std::string> enum_to_string = get_enum_to_string_array<T>();
    // Find index of enum
    auto index_it = std::find(std::begin(enum_to_string),
            std::end(enum_to_string), str);
    // Check if the item exists in the array
    if (index_it == std::end(enum_to_string)) {
        // If it doesn't exist, then print the available choices and throw an
        // exception
        std::stringstream msg;
        msg << std::endl << std::endl << "Oh no! " << str << " is not a valid choice!" << std::endl;
        msg << "Instead, try one of these:" << std::endl;
        for (auto it = std::begin(enum_to_string); it != std::end(enum_to_string); it++) {
            msg << "    " << *it << std::endl;
        }
        throw InputException(msg.str());
    }
    // Get the index and return the proper enum
    auto index = std::distance(std::begin(enum_to_string), index_it);
    return T(index);
}

enum PhysicsType {
    Euler,
    NavierStokes
};
template <>
inline std::vector<std::string> get_enum_to_string_array<PhysicsType> () {
    return std::vector<std::string> {
        "Euler",
        "NavierStokes"
    };
}

enum ICType {
    Uniform,
};
template <>
inline std::vector<std::string> get_enum_to_string_array<ICType> () {
    return std::vector<std::string> {
        "Uniform",
    };
}

enum BasisType {
    LagrangeSeg,
    LagrangeQuad,
    LagrangeHex,
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
inline std::vector<std::string> get_enum_to_string_array<BasisType> () {
    return std::vector<std::string> {
        "LagrangeSeg",
        "LagrangeQuad",
        "LagrangeHex",
        "LegendreSeg",
        "LegendreQuad",
        "LegendreHex",
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
inline std::vector<std::string> get_enum_to_string_array<QuadratureType> () {
    return std::vector<std::string> {
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
inline std::vector<std::string> get_enum_to_string_array<NodeType> () {
    return std::vector<std::string> {
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
inline std::vector<std::string> get_enum_to_string_array<NormType> () {
    return std::vector<std::string> {
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
inline std::vector<std::string> get_enum_to_string_array<ShapeType> () {
    return std::vector<std::string> {
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
inline std::vector<std::string> get_enum_to_string_array<DGSchemeType> () {
    return std::vector<std::string> {
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
inline std::vector<std::string> get_enum_to_string_array<InitType> () {
    return std::vector<std::string> {
        "L2Projection",
        "Interpolation"
    };
}

#endif //DG_ENUMS_H
