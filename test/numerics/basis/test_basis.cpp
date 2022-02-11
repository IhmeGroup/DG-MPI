#include <cstdlib>
#include <math.h>
#include "gtest/gtest.h"
#include "numerics/basis/basis.h"
#include "common/enums.h"
#include "common/defines.h"

constexpr rtype DOUBLE_TOL = 1E-13;
constexpr rtype SINGLE_TOL = 1E-5;
/*
Test the return for the get_num_basis_coeff function for LegendreSeg
*/
TEST(basis_test_suite, test_get_num_basis_coeff_legendreseg){
	for (int order = 0; order < 10; order++){
        Basis::Basis basis(enum_from_string<BasisType>("LegendreSeg"), order);
		int nb = basis.shape.get_num_basis_coeff(order);

		ASSERT_EQ(nb, order + 1);
	}
}

/*
Test the setting of a function pointer for get_1d_nodes
*/
TEST(basis_test_suite, test_get_1d_nodes_pointer_assignment){
    int order = 1;
    Basis::Basis basis(enum_from_string<BasisType>("LagrangeSeg"), order);

    host_view_type_1D xnodes("xnodes", 1);
    basis.get_1d_nodes(0.0, 10.0, 2, xnodes);
    host_view_type_1D  expected("expected", 2);
    expected(0) = 0.0; 
    expected(1) = 10.0;

    for (int i = 0; i < 2; i++){
        EXPECT_NEAR(expected(i), xnodes(i), DOUBLE_TOL);
    }
}

/*
Tests that the Lagrange basis should be nodal (LagrangeSeg)
*/
TEST(basis_test_suite, test_lagrangeseg_basis_should_be_nodal){
    // loop over order (up to p = 6)
    for (int order = 1; order < 6; order++){
        Basis::Basis basis(enum_from_string<BasisType>("LagrangeSeg"), order);

        int nb = order + 1;
        host_view_type_1D xnodes("xnodes", nb);
        host_view_type_2D basis_val("basis_val", nb, nb);
        basis.get_1d_nodes(-1., 1., nb, xnodes);
        host_view_type_2D quad_pts("quad_pts", nb, 1);
        for (int k = 0; k<nb; k++){
            quad_pts(k, 0) = xnodes(k);
        }

        basis.get_values(quad_pts, basis_val);

        host_view_type_2D expected("expected", nb, nb);

        for (int i = 0; i < nb; i++){
            for (int j = 0; j < nb; j++){
                if (i == j){
                    expected(i, j) = 1.;
                }
                EXPECT_NEAR(expected(i, j), basis_val(i, j), DOUBLE_TOL);
            }
        }


    }
}


/*
Tests the Lagrange basis gradient for p=1 (LagrangeSeg)
*/
TEST(basis_test_suite, test_lagrangeseg_basis_gradient_p1){

    int order = 1;
    Basis::Basis basis(enum_from_string<BasisType>("LagrangeSeg"), order);

    int nb = order + 1;
    host_view_type_1D xnodes("xnodes", nb);
    host_view_type_3D basis_grad_ref("basis_grad_ref", nb, nb, 1);
    basis.get_1d_nodes(-1., 1., nb, xnodes);
    host_view_type_2D quad_pts("quad_pts", nb, 1);
    for (int k = 0; k<nb; k++){
        quad_pts(k, 0) = xnodes(k);
    }
    basis.get_grads(quad_pts, basis_grad_ref);

    host_view_type_2D expected("expected", nb, nb);
    expected(0, 0) = -0.5; expected(0, 1) = 0.5;
    expected(1, 0) = -0.5; expected(1, 1) = 0.5;

    for (int i = 0; i < nb; i++){
        for (int j = 0; j < nb; j++){
            EXPECT_NEAR(expected(i, j), basis_grad_ref(i, j, 0), DOUBLE_TOL);
        }
    }
}


/*
Tests that the Lagrange basis should be nodal (LagrangeQuad)
*/
TEST(basis_test_suite, test_lagrangequad_basis_should_be_nodal){
    // loop over order (up to p = 6)
    for (int order = 1; order < 6; order++){
        Basis::Basis basis(enum_from_string<BasisType>("LagrangeQuad"), order);

        int nb = (order + 1.) * (order + 1.);//basis.get_num_basis_coeff(order);

        host_view_type_1D xnodes("xnodes", basis.get_order() + 1);
        host_view_type_2D basis_val("basis_val", nb, nb);
        basis.get_1d_nodes(-1., 1., basis.get_order() + 1, xnodes);
        host_view_type_2D quad_pts("quad_pts", nb, 2);
        
        int nb_1d = basis.get_order() + 1;
        // logic to construct quad pts for a 2d quad
        int dim = 0;
        for (int k = 0; k<nb; k++){
                if (dim % nb_1d == 0) dim = 0;
                quad_pts(k, 0) = xnodes(dim);
                dim += 1;
        }
        dim = 0;
        for (int k = 0; k < nb; k++){
            if (k % nb_1d == 0 && k != 0) dim += 1;
            quad_pts(k, 1) = xnodes(dim);
        }

        basis.get_values(quad_pts, basis_val);

        host_view_type_2D expected("expected", nb, nb);
        for (int i = 0; i < nb; i++){
            for (int j = 0; j < nb; j++){
                if (i == j){
                    expected(i, j) = 1.;
                }
                EXPECT_NEAR(expected(i, j), basis_val(i, j), DOUBLE_TOL);
            }
        }
    }
}

/*
Tests the Lagrange basis gradient for p=1 (LagrangeQuad)
*/
TEST(basis_test_suite, test_lagrangequad_basis_gradient_p1){

    int order = 1;
    Basis::Basis basis(enum_from_string<BasisType>("LagrangeQuad"), order);

    int nb = (order + 1.) * (order + 1.);//basis.get_num_basis_coeff(order);
    int nb_1d = basis.get_order() + 1;
    host_view_type_1D xnodes("xnodes", nb_1d);
    host_view_type_3D  basis_grad_ref("basis_grad_ref", nb, nb, 2);
    basis.get_1d_nodes(-1., 1., nb_1d, xnodes);
    host_view_type_2D  quad_pts("quad_pts", nb, 2);
    
    // logic to construct quad pts for a 2d quad
    int dim = 0;
    for (int k = 0; k<nb; k++){
            if (dim % nb_1d == 0) dim = 0;
            quad_pts(k, 0) = xnodes(dim);
            dim += 1;
    }
    dim = 0;
    for (int k = 0; k < nb; k++){
        if (k % nb_1d == 0 && k != 0) dim += 1;
        quad_pts(k, 1) = xnodes(dim);
    }

    basis.get_grads(quad_pts, basis_grad_ref);

    host_view_type_3D  expected("expected", nb, nb, 2);

    // xi = (-1, -1)
    expected(0, 0, 0) = -.5; expected(0, 0, 1) = -.5;
    expected(0, 1, 0) = .5; expected(0, 1, 1) = 0.;
    expected(0, 2, 0) = 0.; expected(0, 2, 1) = .5;
    expected(0, 3, 0) = 0.; expected(0, 3, 1) = 0.;

    // xi = (1, -1)
    expected(1, 0, 0) = -.5; expected(1, 0, 1) = 0.;
    expected(1, 1, 0) = .5; expected(1, 1, 1) = -.5;
    expected(1, 2, 0) = 0.; expected(1, 2, 1) = 0.;
    expected(1, 3, 0) = 0.; expected(1, 3, 1) = .5;

    // xi = (-1, 1)
    expected(2, 0, 0) = 0.; expected(2, 0, 1) = -.5;
    expected(2, 1, 0) = 0.; expected(2, 1, 1) = 0.;
    expected(2, 2, 0) = -.5; expected(2, 2, 1) = .5;
    expected(2, 3, 0) = .5; expected(2, 3, 1) = 0.;

    // xi = (1, 1)
    expected(3, 0, 0) = 0.; expected(3, 0, 1) = 0.;
    expected(3, 1, 0) = 0.; expected(3, 1, 1) = -.5;
    expected(3, 2, 0) = -.5; expected(3, 2, 1) = 0.;
    expected(3, 3, 0) = .5; expected(3, 3, 1) = .5;

    for (int k = 0; k < 2; k++){
        for (int i = 0; i < nb; i++){
            for (int j = 0; j < nb; j++){
                EXPECT_NEAR(expected(i, j, k), 
                        basis_grad_ref(i, j, k), DOUBLE_TOL);
            }
        }
    }
}

/*
Tests that the Lagrange basis should be nodal (LagrangeHex)
*/
TEST(basis_test_suite, test_lagrangehex_basis_should_be_nodal){
    // loop over order (up to p = 6)
    for (int order = 1; order < 6; order++){
        Basis::Basis basis(enum_from_string<BasisType>("LagrangeHex"), order);

        int nb = basis.shape.get_num_basis_coeff(order);
        int nb_1d = basis.get_order() + 1;
        host_view_type_1D xnodes("xnodes", nb_1d);
        host_view_type_2D basis_val("basis_val", nb, nb);
        basis.get_1d_nodes(-1., 1., nb_1d, xnodes);
        host_view_type_2D quad_pts("quad_pts", nb, 3);
        
        // logic to construct quad pts for a 3d hex
        int dim = 0;
        for (int k = 0; k<nb; k++){
                if (dim % nb_1d == 0) dim = 0;
                quad_pts(k, 0) = xnodes(dim);
                dim += 1;
        }
        dim = 0;
        for (int k = 0; k < nb; k++){
            if (k % nb_1d == 0 && k != 0) dim += 1;
            if (dim == nb_1d) dim = 0;
            quad_pts(k, 1) = xnodes(dim);
        }
        dim = 0;
        int counter = 1;
        for (int k = 0; k < nb; k++){
            if (k == nb_1d*nb_1d*counter) counter +=1;
            quad_pts(k, 2) = xnodes(counter - 1);
        }

        basis.get_values(quad_pts, basis_val);

        host_view_type_2D expected("expected", nb, nb);
        for (int i = 0; i < nb; i++){
            for (int j = 0; j < nb; j++){
                if (i == j){
                    expected(i, j) = 1.;
                }
                EXPECT_NEAR(expected(i, j), basis_val(i, j), DOUBLE_TOL);
            }
        }
    }
}

/*
Tests the Lagrange basis gradient for p=1 (LagrangeHex)
*/
TEST(basis_test_suite, test_lagrangehex_basis_gradient_p1){

    int order = 1;
    Basis::Basis basis(enum_from_string<BasisType>("LagrangeHex"), order);

    int nb = basis.shape.get_num_basis_coeff(order);
    int nb_1d = basis.get_order() + 1;
    host_view_type_1D xnodes("xnodes", nb_1d);
    host_view_type_3D basis_grad_ref("basis_grad_ref", nb, nb, 3);
    basis.get_1d_nodes(-1., 1., nb_1d, xnodes);
    host_view_type_2D quad_pts("quad_pts", 1, 3);
    
    quad_pts(0, 0) = -1.; quad_pts(0, 1) = -1.; quad_pts(0, 2) = -1;

    basis.get_grads(quad_pts, basis_grad_ref);

    host_view_type_3D expected("expected", nb, nb, 3);

    // xi = (-1, -1, -1)
    expected(0, 0, 0) = -.5; expected(0, 0, 1) = -.5; expected(0, 0, 2) = -.5;
    expected(0, 1, 0) = .5;  expected(0, 1, 1) = 0.;  expected(0, 1, 2) = 0.;
    expected(0, 2, 0) = 0.;  expected(0, 2, 1) = .5;  expected(0, 2, 2) = 0.;
    expected(0, 3, 0) = 0.;  expected(0, 3, 1) = 0.;  expected(0, 3, 2) = 0.;
    expected(0, 4, 0) = 0.;  expected(0, 4, 1) = 0.;  expected(0, 4, 2) = .5;
    expected(0, 5, 0) = 0.;  expected(0, 5, 1) = 0.;  expected(0, 5, 2) = 0.;
    expected(0, 6, 0) = 0.;  expected(0, 6, 1) = 0.;  expected(0, 6, 2) = 0.;
    expected(0, 7, 0) = 0.;  expected(0, 7, 1) = 0.;  expected(0, 7, 2) = 0.;


    for (int k = 0; k < 3; k++){
        for (int i = 0; i < 1; i++){
            for (int j = 0; j < nb; j++){
                EXPECT_NEAR(expected(i, j, k), 
                        basis_grad_ref(i, j, k), DOUBLE_TOL);
            }
        }
    }
}

// /*
// Tests the 1D Legendre basis value for P1
// */
// TEST(basis_test_suite, test_legendre_basis_1D_p1){
// 	// allocate data
// 	Kokkos::View<rtype**> quad_pts("quad_pts", 3, 1);
// 	Kokkos::View<rtype**> basis_val("basis_val", 3, 2);
// 	Kokkos::View<rtype**> expected("expected", 3, 2);

// 	expected(0, 0) = 1.; expected(0, 1) = -1.;
// 	expected(1, 0) = 1.; expected(1, 1) = 0.;
// 	expected(2, 0) = 1.; expected(2, 1) = 1.;

// 	int order = 1;
// 	Basis::LegendreSeg basis(order);

// 	quad_pts(0, 0) = -1.; quad_pts(1, 0) = 0.; quad_pts(2, 0) = 1.;

// 	basis.get_values(quad_pts, basis_val);

// 	for (int iq = 0; iq < 3; iq++){
// 		for (int ib = 0; ib < 2; ib++){
// 			EXPECT_NEAR(expected(iq, ib), basis_val(iq, ib), DOUBLE_TOL);
// 		}
// 	}
// }

// /*
// Tests the 1D Legendre basis grad value for P1
// */
// TEST(basis_test_suite, test_legendre_basis_grad_1D_p1){
// 	// allocate data
// 	Kokkos::View<rtype**> quad_pts("quad_pts", 3, 1);
// 	Kokkos::View<rtype***> basis_grad_ref("basis_grad_ref", 3, 2, 1);
// 	Kokkos::View<rtype***> expected("expected", 3, 2, 1);

// 	expected(0, 0, 0) = 0.; expected(0, 1, 0) = 1.;
// 	expected(1, 0, 0) = 0.; expected(1, 1, 0) = 1.;
// 	expected(2, 0, 0) = 0.; expected(2, 1, 0) = 1.;

// 	int order = 1;
// 	Basis::LegendreSeg basis(order);

// 	quad_pts(0, 0) = -1.; quad_pts(1, 0) = 0.; quad_pts(2, 0) = 1.;

// 	basis.get_grads(quad_pts, basis_grad_ref);

// 	for (int iq = 0; iq < 3; iq++){
// 		for (int ib = 0; ib < 2; ib++){
// 			EXPECT_NEAR(expected(iq, ib, 0), basis_grad_ref(iq, ib, 0), DOUBLE_TOL);
// 		}
// 	}

// }

// /* 
// Test of 2D Legendre basis for P4 -> this test is ok... but could use something that 
// is less hard coded. (extracted from the DGLegion code).
// */
// TEST(basis_test_suite, test_legendre_basis_2D_p4) {
//     int order = 4;
    
//     Kokkos::View<rtype**> quad_pts("quad_pts", 2, 2);

//     quad_pts(0, 0) = 0.3;
//     quad_pts(0, 1) = -0.2;
//     quad_pts(1, 0) = 0.7;
//     quad_pts(1, 1) = -0.3;

//     Basis::LegendreQuad basis(order);

//     Kokkos::View<rtype**> basis_val("basis_val", 2, 
//     	basis.get_num_basis_coeff(order));

//     Kokkos::View<rtype***> basis_ref_grad("basis_ref_grad", 2, 
//     	basis.get_num_basis_coeff(order), 2);

//     basis.get_values(quad_pts, basis_val);
//     basis.get_grads(quad_pts, basis_ref_grad);

//     std::vector<rtype> exact_vals_2d{1.000000000000000, 0.3, -0.365, -0.3825, 0.0729375,
// 		-0.2, -0.06, 0.073, 0.0765, -0.0145875,
// 		-0.44, -0.132, 0.1606, 0.1683, -0.0320925,
// 		0.28, 0.084, -0.1022, -0.1071, 0.0204225,
// 		0.232, 0.0696, -0.08468, -0.08874, 0.0169215,
// 		1.000000000000000, 0.7, 0.235, -0.1925, -0.412063,
// 		-0.3, -0.21, -0.0705, 0.05775, 0.123619,
// 		-0.365, -0.2555, -0.085775, 0.0702625, 0.150403,
// 		0.3825, 0.26775, 0.0898875, -0.0736312, -0.157614,
// 		0.0729375, 0.0510563, 0.0171403, -0.0140405, -0.0300548};
   
// 	std::vector<rtype> exact_grads_2d{0, 1.000000000000000, 0.9, -0.825, -1.7775,
// 		0, -0.2, -0.18, 0.165, 0.3555,
// 		0, -0.44, -0.396, 0.363, 0.7821,
// 		0, 0.28, 0.252, -0.231, -0.4977,
// 		0, 0.232, 0.2088, -0.1914, -0.41238,
// 		0, 1.000000000000000, 2.1, 2.175, 0.7525,
// 		0, -0.3, -0.63, -0.6525, -0.22575,
// 		0, -0.365, -0.7665, -0.793875, -0.274662,
// 		0, 0.3825, 0.80325, 0.831937, 0.287831,
// 		0, 0.0729375, 0.153169, 0.158639, 0.0548855,
// 		0, 0, 0, 0, 0,
// 		1.000000000000000, 0.3, -0.365, -0.3825,
// 		0.0729375,
// 		-0.6, -0.18, 0.219, 0.2295, -0.0437625,
// 		-1.2, -0.36, 0.438, 0.459, -0.087525,
// 		1.36, 0.408, -0.4964, -0.5202, 0.099195,
// 		0, 0, 0, 0, 0,
// 		1.000000000000000, 0.7, 0.235, -0.1925, -0.412063,
// 		-0.9, -0.63, -0.2115, 0.17325, 0.370856,
// 		-0.825, -0.5775, -0.193875, 0.158813, 0.339952,
// 		1.7775, 1.24425, 0.417712, -0.342169, -0.732441};

//     for (int iq = 0; iq < 2; iq++){
//     	for (int ib = 0; ib < 25; ib++){
//     		EXPECT_NEAR(basis_val(iq, ib), 
//     				exact_vals_2d[ib + iq * 25], SINGLE_TOL);
//     	}
//     }                 
//     for (int idim = 0; idim < 2; idim++){
//     	for (int iq = 0; iq < 2; iq++) {
//     		for (int ib = 0; ib < 25; ib++){
//     			EXPECT_NEAR(basis_ref_grad(iq, ib, idim), 
//     				exact_grads_2d[ib + iq * 25 + idim * 25 * 2], SINGLE_TOL);
//     		}
//     	}
//     }           
// }

// /* 
// Test of 3D Legendre basis for P4 -> this test is ok... but could use something that 
// is less hard coded. (extracted from the DGLegion code).
// */
// TEST(basis_test_suite, test_legendre_basis_3D_p4) {
//     int order = 4;
    
//     Kokkos::View<rtype**> quad_pts("quad_pts", 2, 3);
//     quad_pts(0, 0) = 0.3;
//     quad_pts(0, 1) = -0.2;
//     quad_pts(0, 2) = 0.5;
//     quad_pts(1, 0) = 0.7;
//     quad_pts(1, 1) = -0.3;
//     quad_pts(1, 2) = 0.1; 

//     Basis::LegendreHex basis(order);

//     Kokkos::View<rtype**> basis_val("basis_val", 2, 
//     	basis.get_num_basis_coeff(order));

//     Kokkos::View<rtype***> basis_ref_grad("basis_ref_grad", 2, 
//     	basis.get_num_basis_coeff(order), 3);

//     basis.get_values(quad_pts, basis_val);
//     basis.get_grads(quad_pts, basis_ref_grad);

//     std::vector<rtype> exact_vals_3d{
//         1.000000000000000, 0.3, -0.365, -0.3825, 0.0729375,
//         -0.2, -0.06, 0.073, 0.0765, -0.0145875,
//         -0.44, -0.132, 0.1606, 0.1683, -0.0320925,
//         0.28, 0.084, -0.1022, -0.1071, 0.0204225,
//         0.232, 0.0696, -0.08468, -0.08874, 0.0169215,
//         0.5, 0.15, -0.1825, -0.19125, 0.0364688,
//         -0.1, -0.03, 0.0365, 0.03825, -0.00729375,
//         -0.22, -0.066, 0.0803, 0.08415, -0.0160463,
//         0.14, 0.042, -0.0511, -0.05355, 0.0102113,
//         0.116, 0.0348, -0.04234, -0.04437, 0.00846075,
//         -0.125, -0.0375, 0.045625, 0.0478125, -0.00911719,
//         0.025, 0.0075, -0.009125, -0.0095625, 0.00182344,
//         0.055, 0.0165, -0.020075, -0.0210375, 0.00401156,
//         -0.035, -0.0105, 0.012775, 0.0133875, -0.00255281,
//         -0.029, -0.0087, 0.010585, 0.0110925, -0.00211519,
//         -0.4375, -0.13125, 0.159688, 0.167344, -0.0319102,
//         0.0875, 0.02625, -0.0319375, -0.0334688, 0.00638203,
//         0.1925, 0.05775, -0.0702625, -0.0736312, 0.0140405,
//         -0.1225, -0.03675, 0.0447125, 0.0468562, -0.00893484,
//         -0.1015, -0.03045, 0.0370475, 0.0388237, -0.00740316,
//         -0.289063, -0.0867188, 0.105508, 0.110566, -0.0210835,
//         0.0578125, 0.0173438, -0.0211016, -0.0221133, 0.0042167,
//         0.127188, 0.0381563, -0.0464234, -0.0486492, 0.00927674,
//         -0.0809375, -0.0242813, 0.0295422, 0.0309586, -0.00590338,
//         -0.0670625, -0.0201188, 0.0244778, 0.0256514, -0.00489137,

//         1.000000000000000, 0.7, 0.235, -0.1925, -0.412063,
//         -0.3, -0.21, -0.0705, 0.05775, 0.123619,
//         -0.365, -0.2555, -0.085775, 0.0702625, 0.150403,
//         0.3825, 0.26775, 0.0898875, -0.0736312, -0.157614,
//         0.0729375, 0.0510563, 0.0171403, -0.0140405, -0.0300548,
//         0.1, 0.07, 0.0235, -0.01925, -0.0412063,
//         -0.03, -0.021, -0.00705, 0.005775, 0.0123619,
//         -0.0365, -0.02555, -0.0085775, 0.00702625, 0.0150403,
//         0.03825, 0.026775, 0.00898875, -0.00736312, -0.0157614,
//         0.00729375, 0.00510563, 0.00171403, -0.00140405, -0.00300548,
//         -0.485, -0.3395, -0.113975, 0.0933625, 0.19985,
//         0.1455, 0.10185, 0.0341925, -0.0280088, -0.0599551,
//         0.177025, 0.123918, 0.0416009, -0.0340773, -0.0729454,
//         -0.185513, -0.129859, -0.0435954, 0.0357112, 0.0764427,
//         -0.0353747, -0.0247623, -0.00831305, 0.00680963, 0.0145766,
//         -0.1475, -0.10325, -0.0346625, 0.0283938, 0.0607792,
//         0.04425, 0.030975, 0.0103987, -0.00851813, -0.0182338,
//         0.0538375, 0.0376863, 0.0126518, -0.0103637, -0.0221844,
//         -0.0564188, -0.0394931, -0.0132584, 0.0108606, 0.0232481,
//         -0.0107583, -0.0075308, -0.0025282, 0.00207097, 0.00443308,
//         0.337938, 0.236556, 0.0794153, -0.065053, -0.139251,
//         -0.101381, -0.0709669, -0.0238246, 0.0195159, 0.0417754,
//         -0.123347, -0.086343, -0.0289866, 0.0237443, 0.0508268,
//         0.129261, 0.0904828, 0.0303764, -0.0248828, -0.0532636,
//         0.0246483, 0.0172538, 0.00579235, -0.0047448, -0.0101566
//     };
   
//     std::vector<rtype> exact_grads_3d{
//         0, 1.000000000000000, 0.9, -0.825, -1.7775,
//         0, -0.2, -0.18, 0.165, 0.3555,
//         0, -0.44, -0.396, 0.363, 0.7821,
//         0, 0.28, 0.252, -0.231, -0.4977,
//         0, 0.232, 0.2088, -0.1914, -0.41238,
//         0, 0.5, 0.45, -0.4125, -0.88875,
//         0, -0.1, -0.09, 0.0825, 0.17775,
//         0, -0.22, -0.198, 0.1815, 0.39105,
//         0, 0.14, 0.126, -0.1155, -0.24885,
//         0, 0.116, 0.1044, -0.0957, -0.20619,
//         0, -0.125, -0.1125, 0.103125, 0.222188,
//         0, 0.025, 0.0225, -0.020625, -0.0444375,
//         0, 0.055, 0.0495, -0.045375, -0.0977625,
//         0, -0.035, -0.0315, 0.028875, 0.0622125,
//         0, -0.029, -0.0261, 0.023925, 0.0515475,
//         0, -0.4375, -0.39375, 0.360938, 0.777656,
//         0, 0.0875, 0.07875, -0.0721875, -0.155531,
//         0, 0.1925, 0.17325, -0.158813, -0.342169,
//         0, -0.1225, -0.11025, 0.101063, 0.217744,
//         0, -0.1015, -0.09135, 0.0837375, 0.180416,
//         0, -0.289063, -0.260156, 0.238477, 0.513809,
//         0, 0.0578125, 0.0520313, -0.0476953, -0.102762,
//         0, 0.127188, 0.114469, -0.10493, -0.226076,
//         0, -0.0809375, -0.0728438, 0.0667734, 0.143866,
//         0, -0.0670625, -0.0603562, 0.0553266, 0.119204,

//         0, 1.000000000000000, 2.1, 2.175, 0.7525,
//         0, -0.3, -0.63, -0.6525, -0.22575,
//         0, -0.365, -0.7665, -0.793875, -0.274662,
//         0, 0.3825, 0.80325, 0.831937, 0.287831,
//         0, 0.0729375, 0.153169, 0.158639, 0.0548855,
//         0, 0.1, 0.21, 0.2175, 0.07525,
//         0, -0.03, -0.063, -0.06525, -0.022575,
//         0, -0.0365, -0.07665, -0.0793875, -0.0274662,
//         0, 0.03825, 0.080325, 0.0831937, 0.0287831,
//         0, 0.00729375, 0.0153169, 0.0158639, 0.00548855,
//         0, -0.485, -1.0185, -1.05488, -0.364962,
//         0, 0.1455, 0.30555, 0.316462, 0.109489,
//         0, 0.177025, 0.371752, 0.385029, 0.133211,
//         0, -0.185513, -0.389576, -0.40349, -0.139598,
//         0, -0.0353747, -0.0742868, -0.0769399, -0.0266195,
//         0, -0.1475, -0.30975, -0.320813, -0.110994,
//         0, 0.04425, 0.092925, 0.0962437, 0.0332981,
//         0, 0.0538375, 0.113059, 0.117097, 0.0405127,
//         0, -0.0564188, -0.118479, -0.122711, -0.0424551,
//         0, -0.0107583, -0.0225924, -0.0233993, -0.00809561,
//         0, 0.337938, 0.709669, 0.735014, 0.254298,
//         0, -0.101381, -0.212901, -0.220504, -0.0762894,
//         0, -0.123347, -0.259029, -0.26828, -0.0928188,
//         0, 0.129261, 0.271448, 0.281143, 0.097269,
//         0, 0.0246483, 0.0517615, 0.0536101, 0.0185479,

//         0, 0, 0, 0, 0,
//         1.000000000000000, 0.3, -0.365, -0.3825, 0.0729375,
//         -0.6, -0.18, 0.219, 0.2295, -0.0437625,
//         -1.2, -0.36, 0.438, 0.459, -0.087525,
//         1.36, 0.408, -0.4964, -0.5202, 0.099195,
//         0, 0, 0, 0, 0,
//         0.5, 0.15, -0.1825, -0.19125, 0.0364688,
//         -0.3, -0.09, 0.1095, 0.11475, -0.0218813,
//         -0.6, -0.18, 0.219, 0.2295, -0.0437625,
//         0.68, 0.204, -0.2482, -0.2601, 0.0495975,
//         0, 0, 0, 0, 0,
//         -0.125, -0.0375, 0.045625, 0.0478125, -0.00911719,
//         0.075, 0.0225, -0.027375, -0.0286875, 0.00547031,
//         0.15, 0.045, -0.05475, -0.057375, 0.0109406,
//         -0.17, -0.051, 0.06205, 0.065025, -0.0123994,
//         0, 0, 0, 0, 0,
//         -0.4375, -0.13125, 0.159688, 0.167344, -0.0319102,
//         0.2625, 0.07875, -0.0958125, -0.100406, 0.0191461,
//         0.525, 0.1575, -0.191625, -0.200813, 0.0382922,
//         -0.595, -0.1785, 0.217175, 0.227588, -0.0433978,
//         0, 0, 0, 0, 0,
//         -0.289063, -0.0867188, 0.105508, 0.110566, -0.0210835,
//         0.173438, 0.0520313, -0.0633047, -0.0663398, 0.0126501,
//         0.346875, 0.104063, -0.126609, -0.13268, 0.0253002,
//         -0.393125, -0.117938, 0.143491, 0.15037, -0.0286736,

//         0, 0, 0, 0, 0,
//         1.000000000000000, 0.7, 0.235, -0.1925, -0.412063,
//         -0.9, -0.63, -0.2115, 0.17325, 0.370856,
//         -0.825, -0.5775, -0.193875, 0.158813, 0.339952,
//         1.7775, 1.24425, 0.417712, -0.342169, -0.732441,
//         0, 0, 0, 0, 0,
//         0.1, 0.07, 0.0235, -0.01925, -0.0412063,
//         -0.09, -0.063, -0.02115, 0.017325, 0.0370856,
//         -0.0825, -0.05775, -0.0193875, 0.0158813, 0.0339952,
//         0.17775, 0.124425, 0.0417712, -0.0342169, -0.0732441,
//         0, 0, 0, 0, 0,
//         -0.485, -0.3395, -0.113975, 0.0933625, 0.19985,
//         0.4365, 0.30555, 0.102577, -0.0840262, -0.179865,
//         0.400125, 0.280088, 0.0940294, -0.0770241, -0.164877,
//         -0.862088, -0.603461, -0.202591, 0.165952, 0.355234,
//         0, 0, 0, 0, 0,
//         -0.1475, -0.10325, -0.0346625, 0.0283938, 0.0607792,
//         0.13275, 0.092925, 0.0311962, -0.0255544, -0.0547013,
//         0.121688, 0.0851813, 0.0285966, -0.0234248, -0.0501429,
//         -0.262181, -0.183527, -0.0616126, 0.0504699, 0.108035,
//         0, 0, 0, 0, 0,
//         0.337938, 0.236556, 0.0794153, -0.065053, -0.139251,
//         -0.304144, -0.212901, -0.0714738, 0.0585477, 0.125326,
//         -0.278798, -0.195159, -0.0655176, 0.0536687, 0.114882,
//         0.600684, 0.420479, 0.141161, -0.115632, -0.247519,

//         0, 0, 0, 0, 0,
//         0, 0, 0, 0, 0,
//         0, 0, 0, 0, 0,
//         0, 0, 0, 0, 0,
//         0, 0, 0, 0, 0,
//             1.000000000000000, 0.3, -0.365, -0.3825, 0.0729375,
//         -0.2, -0.06, 0.073, 0.0765, -0.0145875,
//         -0.44, -0.132, 0.1606, 0.1683, -0.0320925,
//         0.28, 0.084, -0.1022, -0.1071, 0.0204225,
//         0.232, 0.0696, -0.08468, -0.08874, 0.0169215,
//             1.5, 0.45, -0.5475, -0.57375, 0.109406,
//         -0.3, -0.09, 0.1095, 0.11475, -0.0218813,
//         -0.66, -0.198, 0.2409, 0.25245, -0.0481388,
//         0.42, 0.126, -0.1533, -0.16065, 0.0306338,
//         0.348, 0.1044, -0.12702, -0.13311, 0.0253823,
//             0.375, 0.1125, -0.136875, -0.143438, 0.0273516,
//         -0.075, -0.0225, 0.027375, 0.0286875, -0.00547031,
//         -0.165, -0.0495, 0.060225, 0.0631125, -0.0120347,
//         0.105, 0.0315, -0.038325, -0.0401625, 0.00765844,
//         0.087, 0.0261, -0.031755, -0.0332775, 0.00634556,
//             -1.5625, -0.46875, 0.570313, 0.597656, -0.113965,
//         0.3125, 0.09375, -0.114063, -0.119531, 0.022793,
//         0.6875, 0.20625, -0.250938, -0.262969, 0.0501445,
//         -0.4375, -0.13125, 0.159688, 0.167344, -0.0319102,
//         -0.3625, -0.10875, 0.132313, 0.138656, -0.0264398,

//         0, 0, 0, 0, 0,
//         0, 0, 0, 0, 0,
//         0, 0, 0, 0, 0,
//         0, 0, 0, 0, 0,
//         0, 0, 0, 0, 0,
//         1.000000000000000, 0.7, 0.235, -0.1925, -0.412063,
//         -0.3, -0.21, -0.0705, 0.05775, 0.123619,
//         -0.365, -0.2555, -0.085775, 0.0702625, 0.150403,
//         0.3825, 0.26775, 0.0898875, -0.0736312, -0.157614,
//         0.0729375, 0.0510563, 0.0171403, -0.0140405, -0.0300548,
//         0.3, 0.21, 0.0705, -0.05775, -0.123619,
//         -0.09, -0.063, -0.02115, 0.017325, 0.0370856,
//         -0.1095, -0.07665, -0.0257325, 0.0210788, 0.0451208,
//         0.11475, 0.080325, 0.0269662, -0.0220894, -0.0472842,
//         0.0218813, 0.0153169, 0.00514209, -0.00421214, -0.00901644,
//         -1.425, -0.9975, -0.334875, 0.274313, 0.587189,
//         0.4275, 0.29925, 0.100462, -0.0822937, -0.176157,
//         0.520125, 0.364087, 0.122229, -0.100124, -0.214324,
//         -0.545062, -0.381544, -0.12809, 0.104925, 0.2246,
//         -0.103936, -0.0727552, -0.0244249, 0.0200077, 0.0428281,
//         -0.7325, -0.51275, -0.172137, 0.141006, 0.301836,
//         0.21975, 0.153825, 0.0516412, -0.0423019, -0.0905507,
//         0.267363, 0.187154, 0.0628302, -0.0514673, -0.11017,
//         -0.280181, -0.196127, -0.0658426, 0.0539349, 0.115452,
//         -0.0534267, -0.0373987, -0.0125553, 0.0102846, 0.0220151
//     };

//     for (int iq = 0; iq < 2; iq++){
//     	for (int ib = 0; ib < 125; ib++){
//     		EXPECT_NEAR(basis_val(iq, ib), 
//     				exact_vals_3d[ib + iq * 125], SINGLE_TOL);
//     	}
//     }                 
//     for (int idim = 0; idim < 2; idim++){
//     	for (int iq = 0; iq < 2; iq++) {
//     		for (int ib = 0; ib < 125; ib++){
//     			EXPECT_NEAR(basis_ref_grad(iq, ib, idim), 
//     				exact_grads_3d[ib + iq * 125 + idim * 125 * 2], SINGLE_TOL);
//     		}
//     	}
//     }           
// }