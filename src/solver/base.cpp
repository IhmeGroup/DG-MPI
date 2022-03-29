#include "solver/base.h"
#include "solver/helpers.h"
#include "solver/tools.h"
#include "numerics/basis/basis.h"
#include "numerics/basis/tools.h"

#include "H5Cpp.h"

#include "common/defines.h"
#include <iostream>

// #include "physics/euler/functions.h"
// #include "physics/euler/data.h"


using namespace VolumeHelpers;

template<unsigned dim>
Solver<dim>::Solver(const toml::value &input_file, Mesh& mesh, MemoryNetwork& network,
    Numerics::NumericsParams& params, PhysicsType physics_type)
    : input_file{input_file}, mesh{mesh}, network{network}, params{params} {

    // initialize time to zero
    time = 0.0;
    const auto IC_name = toml::find<std::string>(input_file, "InitialCondition", "name");

    order = toml::find<int>(input_file, "Numerics", "order");
    
    // instantiate the basis class
    basis = Basis::Basis(params.basis, order);
    // instantiate the physics class
    physics = Physics::Physics<dim>(physics_type, IC_name);

    stepper = std::shared_ptr<StepperBase<dim>>(new FE<dim>());

    // set the size and shape of the solution coefficient views
    Kokkos::resize(Uc, mesh.num_elems_part, basis.get_num_basis_coeffs(), physics.get_NS());
    h_Uc = Kokkos::create_mirror_view(Uc);

    Kokkos::resize(U_face, mesh.num_ifaces_part, basis.get_num_basis_coeffs(), physics.get_NS());

    // set the size and shape of the residuals view
    Kokkos::resize(res, mesh.num_elems_part, basis.get_num_basis_coeffs(), physics.get_NS());

}

template<unsigned dim>
void Solver<dim>::precompute_matrix_helpers() {

    // ---------------------------------------------------------------------------------------
    //                          Volume Helpers
    // ---------------------------------------------------------------------------------------
    // need to get the sizes of things to pass into scratch memory
    int nb = basis.get_num_basis_coeffs();
    int nq_1d; int nq;
    int qorder = basis.shape.get_quadrature_order(2 * basis.get_order());
    QuadratureTools::get_number_of_quadrature_points(qorder, mesh.dim,
            nq_1d, nq);

    // set scratch memory size for iMM and volume helpers
    int scratch_size_iMM = scratch_view_2D_rtype::shmem_size(mesh.num_nodes_per_elem, mesh.dim)
        + scratch_view_2D_rtype::shmem_size(nb, nb) + scratch_view_2D_rtype::shmem_size(nq, nb);
    int scratch_size_vol = scratch_view_2D_rtype::shmem_size(mesh.num_nodes_per_elem, mesh.dim);

    printf("##### Construct Inverse Mass Matrices #####\n");
    vol_helpers.compute_inv_mass_matrices(scratch_size_iMM, mesh, basis);
    Kokkos::fence();
    printf("##### Completed #####\n");
    printf("##### Construct Volume Helpers #####\n");
    vol_helpers.compute_volume_helpers(scratch_size_vol, mesh, basis);
    Kokkos::fence();
    printf("##### Completed #####\n");


    // ---------------------------------------------------------------------------------------
    //                          Interior Face Helpers
    // ---------------------------------------------------------------------------------------
    
    int scratch_size = 0; // TODO: Will need scratch space for ijac evaluated at faces
    printf("##### Construct Face Helpers #####\n");
    iface_helpers.compute_interior_face_helpers(scratch_size, mesh, basis);
    printf("##### Completed #####\n");


}

template<unsigned dim>
void Solver<dim>::init_state_from_fcn(Mesh& mesh_local){

    // The current initialization assumes that we are using an L2 projection for the 
    // initial conditions.

    // we also pass in a local reference of mesh even though it is a part of the solver
    // object. I believe this has to do with mesh being defined by reference in solver
    // It may need to be stored by value in solver...
    
    // Over-integrate for the initial conditions -> compute quadrature
    int NDIMS = basis.shape.get_NDIMS();
    int nb = basis.get_num_basis_coeffs();

    int nq_1d; int nq;
    int qorder = basis.shape.get_quadrature_order(2*order);
    QuadratureTools::get_number_of_quadrature_points(qorder, NDIMS,
            nq_1d, nq);

    printf("nq=%i\n", nq);
    printf("NDIMS=%i\n", NDIMS);
    view_type_2D quad_pts("quad_pts_ICs", nq, NDIMS);
    view_type_1D quad_wts("quad_wts_ICs", nq);
    host_view_type_2D h_quad_pts = Kokkos::create_mirror_view(quad_pts);
    host_view_type_1D h_quad_wts = Kokkos::create_mirror_view(quad_wts);

    basis.shape.get_quadrature_data(qorder, nq_1d, h_quad_pts, h_quad_wts);
    Kokkos::deep_copy(quad_pts, h_quad_pts);
    Kokkos::deep_copy(quad_wts, h_quad_wts);

    // Get the basis values evaluated at the 
    // overintegrated quadrature points
    view_type_2D basis_val("basis_val_ICs", nq, nb);
    view_type_2D gbasis_val("gbasis_val_ICs", nq, nb);
    host_view_type_2D h_basis_val = Kokkos::create_mirror_view(basis_val);
    host_view_type_2D h_gbasis_val = Kokkos::create_mirror_view(gbasis_val);

    basis.get_values(h_quad_pts, h_basis_val);
    mesh.gbasis.get_values(h_quad_pts, h_gbasis_val);

    Kokkos::deep_copy(basis_val, h_basis_val);
    Kokkos::deep_copy(gbasis_val, h_gbasis_val);

    // Get the geometric basis ref gradient evaluated
    // at the overintegrated quadrature points
    view_type_3D gbasis_ref_grad("gbasis_ref_grad_ICs", nq, nb, NDIMS);
    host_view_type_3D h_gbasis_ref_grad = Kokkos::create_mirror_view(gbasis_ref_grad);

    mesh.gbasis.get_grads(h_quad_pts, h_gbasis_ref_grad);
    Kokkos::deep_copy(gbasis_ref_grad, h_gbasis_ref_grad);

    // set the scratch size for the parallel_for loop
    int scratch_size = scratch_view_2D_rtype::shmem_size(mesh.num_nodes_per_elem, mesh.dim)
        + scratch_view_2D_rtype::shmem_size(nq, NDIMS) 
        + scratch_view_2D_rtype::shmem_size(nq, physics.get_NS())
        + scratch_view_1D_rtype::shmem_size(nq)
        + scratch_view_3D_rtype::shmem_size(nq, NDIMS, NDIMS)
        + scratch_view_2D_rtype::shmem_size(nb, physics.get_NS());

    Kokkos::parallel_for("init state", Kokkos::TeamPolicy<>( mesh.num_elems_part,
        Kokkos::AUTO).set_scratch_size( 1, Kokkos::PerTeam( scratch_size )),
        KOKKOS_CLASS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& member) {

            const int elem_ID = member.league_rank();

            // set the scratch memory for local memory
            scratch_view_2D_rtype elem_coords(member.team_scratch( 1 ),
                mesh_local.num_nodes_per_elem, NDIMS);

            scratch_view_2D_rtype xphys(member.team_scratch( 1 ),
                nq, NDIMS);

            scratch_view_2D_rtype f(member.team_scratch( 1 ),
                nq, physics.NUM_STATE_VARS);

            scratch_view_1D_rtype djac(member.team_scratch( 1 ),
                nq);

            scratch_view_3D_rtype jac(member.team_scratch( 1 ),
                nq, NDIMS, NDIMS);

            // get the physical location of the evaluation points
            if (member.team_rank() == 0 ) {
                MeshTools::elem_coords_from_elem_ID(mesh_local, elem_ID, elem_coords,
                    member);
                MeshTools::ref_to_phys(gbasis_val, xphys, elem_coords, member);            
            }
            member.team_barrier();

            // Evaluate the initial condition at the evaluated coordinates in physical space
            Kokkos::parallel_for(Kokkos::TeamThreadRange(member, nq), [&] (const int iq) {
                physics.call_IC(Kokkos::subview(xphys, iq, Kokkos::ALL()), time,
                    Kokkos::subview(f, iq, Kokkos::ALL()));
            });
            member.team_barrier();

            // Evaluate djac
            BasisTools::get_element_jacobian(quad_pts, gbasis_ref_grad,
                jac, djac, elem_coords, member);
            member.team_barrier();


            // L2 projection
            SolverTools::L2_projection(
                Kokkos::subview(vol_helpers.iMM_elems, elem_ID, Kokkos::ALL(), Kokkos::ALL()),
                basis_val, djac, quad_wts, f,
                Kokkos::subview(Uc, elem_ID, Kokkos::ALL(), Kokkos::ALL()), member);

        });


    // for (unsigned long i=0; i<Uc.extent(0); i++){
    //     for (unsigned long j=0; j<Uc.extent(1); j++){
    //         for (unsigned long k=0; k<Uc.extent(2); k++){
    //             printf("Uc(%i, %i, %i)=%f\n", i, j, k, Uc(i,j,k));
    //         }
    //     }
    // }
}
template<unsigned dim>
void Solver<dim>::copy_from_device_to_host(){
    Kokkos::deep_copy(h_Uc, Uc);
}

template<unsigned dim>
void Solver<dim>::read_in_coefficients(const std::string& filename){
    // This function reads in the solution coefficients from an 
    // hdf5 file. It detects whether the data is stored as 
    // column or row major prior to reading it in. 

    // TODO: This implementation will likely change once we
    //       begin using parallel hdf5

    // loop over each rank
    for (unsigned rank = 0; rank < network.num_ranks; rank++){

        network.barrier();
        //Preform this in serial
        if (rank == network.rank){  

            H5::H5File file(filename, H5F_ACC_RDONLY);
            hsize_t dims[3]; // buffer to store an HDF5 dataset dimensions

            int num_elems_part;
            int nb;
            int ns;
            bool stored_layout;

            // add head group with attributes
            auto group = file.openGroup("/"); 

            auto attr = group.openAttribute("Number of Basis Functions");
            auto type = attr.getDataType();
            attr.read(type, &nb);

            attr = group.openAttribute("Number of State Variables");
            attr.read(type, &ns);


            // read in solver coefficients
            group = file.openGroup("Rank "+ std::to_string(network.rank));

            // number of elements
            attr = group.openAttribute("Number of Elements per Partition");
            type = attr.getDataType();
            attr.read(type, &num_elems_part);

            // Row (LayoutRight) vs Column (LayoutLeft) Major Flag
            // Note: Row = True Column = False
            attr = group.openAttribute("Stored Layout");
            type = attr.getDataType();
            attr.read(type, &stored_layout);

            vector<rtype> buff(num_elems_part * nb * ns);
            dims[0] = num_elems_part;
            dims[1] = nb;
            dims[2] = ns;

            DataSpace mspace(3, dims);

            auto dataset = group.openDataSet("Solution Coefficients");
            auto dataspace = dataset.getSpace();

            dataset.read(buff.data(), PredType::NATIVE_DOUBLE, mspace, dataspace);

            Kokkos::resize(Uc, num_elems_part, nb, ns);
            host_view_type_3D h_Uc = Kokkos::create_mirror_view(Uc);

            if (stored_layout == true){
                // read in row major (LayoutRight)
                for (unsigned i = 0; i < num_elems_part; i++){
                    for (unsigned j = 0; j < nb; j++){
                        for (unsigned k = 0; k < ns; k++){
                            h_Uc(i, j, k) = buff[i * nb * ns + j * ns + k];
                        }
                    }
                }
            } else {
                // read in column major (LayoutLeft)
                for (unsigned i = 0; i < ns; i++){
                    for (unsigned j = 0; j < nb; j++){
                        for (unsigned k = 0; k < num_elems_part; k++){
                            h_Uc(k, j, i) = buff[i * nb * num_elems_part + j * num_elems_part + k];
                        }
                    }
                }
            }

            Kokkos::deep_copy(Uc, h_Uc);

            file.close();
        } // end if statement (rank == network.rank)
    } // end loop over ranks
}


template<unsigned dim>
void Solver<dim>::solve(){

    // unpack solver if needed

    // this can be where we place parameters for writing data
    
    // advance the time step
    stepper->take_time_step(*this);

}

template<unsigned dim>
void Solver<dim>::get_residual(){

    get_element_residuals();

}

template<unsigned dim>
void Solver<dim>::get_element_residuals(){

    // unpack
    auto basis_val = vol_helpers.basis_val;


    // allocate state evaluated at quadrature points
    view_type_3D Uq("Uq", mesh.num_elems_part,
        basis_val.extent(0), physics.get_NS());

    Kokkos::fence();
    
    // Evaluate the state
    VolumeHelpers::evaluate_state(mesh.num_elems_part,
        basis_val, Uc, Uq);
    Kokkos::fence();

    // TODO: Evaluate gradient of state at quad points if needed
    printf("I made it here\n");

}

template class Solver<2>;
template class Solver<3>;