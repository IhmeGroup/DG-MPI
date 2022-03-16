#include <algorithm>
#include <iostream>
#include <map>
#include <numeric>
#include <string>
#include <unordered_set>
#include <set>
#include "H5Cpp.h"
#include "metis.h"
#include "toml11/toml.hpp"
#include "common/my_exceptions.h"

using namespace std;
using H5::PredType, H5::DataSpace, H5::DataSet;
using std::cout, std::endl;

// identifiers from Eric's pre-processing tool
static const string DSET_DIM("Dimension");
static const string DSET_QORDER("QOrder");
static const string DSET_NELEM("nElem");
static const string DSET_NNODE("nNode");
static const string DSET_NIFACE("nIFace");
static const string DSET_NNODE_PER_ELEM("nNodePerElem");
static const string DSET_NNODE_PER_FACE("nNodePerFace");
static const string DSET_NODE_COORD("NodeCoords");
static const string DSET_ELEM_TO_NODES("Elem2Nodes");
static const string DSET_IFACE_DATA("IFaceData");

inline Mesh::Mesh(const toml::value &input_info, unsigned num_ranks, unsigned
        rank, bool head_rank, Basis::Basis gbasis, string mesh_file_name) :
        gbasis{gbasis}, num_ranks{num_ranks}, rank{rank}, head_rank{head_rank} {
    auto mesh_info = toml::find(input_info, "Mesh");
    // If the mesh file name is not specified, then read it from the input file
    if (mesh_file_name == "") {
        mesh_file_name = toml::find<string>(mesh_info, "file");
    }

    // If the number of partitions is specified, use that
    if (input_info.contains("npartitions")) {
        num_partitions = toml::find<int>(mesh_info, "npartitions");
    // Otherwise, the number of ranks is used
    } else {
        num_partitions = num_ranks;
    }

    // TODO figure out boundaries from the HDF5 file directly (Kihiro 2021/03/04)
    if (input_info.contains("Boundaries")) {
        BFGnames = toml::find<vector<string>>(input_info, "Boundaries", "names");
    }
    else {
        nBFG = 0;
        nBF = 0;
    }

    // Read in the mesh
    read_mesh(mesh_file_name);
    // partition the mesh using METIS
    partition();
    // Print report on the head rank
    if (head_rank) {
        cout << report() << endl;
    }
}

inline void Mesh::finalize() {
    // Make sure all threads are here (and therefore done with this data) before
    // destroying it
    Kokkos::fence();
    for (unsigned i = 0; i < num_neighbor_ranks; i++) {
        // Explicitly destruct inner views to avoid memory leak
        using Kokkos::View;
        ghost_faces[i].~View<unsigned*>();
    }
    // TODO: This delete has been commented out, because it messed with
    // Kokkos's reference counts of View pointers. Yes, it's a memory leak for
    // now. It should be a small one.
    //delete [] ghost_faces;
}

inline void Mesh::read_mesh(const string &mesh_file_name) {
    try {
        H5::H5File file(mesh_file_name, H5F_ACC_RDONLY);
        hsize_t dims[2]; // buffer to store an HDF5 dataset dimensions
        unsigned rank; // the number of dimensions in a dataset

        dims[0] = 1; // fetch all scalars first
        rank = 1; // rank is the number of dimensions
        DataSpace mspace(rank, dims);

        // number of spatial dimensions
        DataSet dataset = file.openDataSet(DSET_DIM);
        DataSpace dataspace = dataset.getSpace();
        dataset.read(&dim, PredType::NATIVE_INT, mspace, dataspace);
        // geometric order of the mesh
        dataset = file.openDataSet(DSET_QORDER);
        dataspace = dataset.getSpace();
        dataset.read(&order, PredType::NATIVE_INT, mspace, dataspace);
        // number of elements
        dataset = file.openDataSet(DSET_NELEM);
        dataspace = dataset.getSpace();
        dataset.read(&num_elems, PredType::NATIVE_INT, mspace, dataspace);
        // number of nodes
        dataset = file.openDataSet(DSET_NNODE);
        dataspace = dataset.getSpace();
        dataset.read(&num_nodes, PredType::NATIVE_INT, mspace, dataspace);
        // number of interior faces
        dataset = file.openDataSet(DSET_NIFACE);
        dataspace = dataset.getSpace();
        dataset.read(&nIF, PredType::NATIVE_INT, mspace, dataspace);
        // number of nodes per element
        dataset = file.openDataSet(DSET_NNODE_PER_ELEM);
        dataspace = dataset.getSpace();
        dataset.read(&num_nodes_per_elem, PredType::NATIVE_INT, mspace, dataspace);
        // number of nodes per faces
        if (H5Lexists(file.getId(), DSET_NNODE_PER_FACE.c_str(), H5P_DEFAULT)) {
            dataset = file.openDataSet(DSET_NNODE_PER_FACE);
            dataspace = dataset.getSpace();
            dataset.read(&num_nodes_per_face, PredType::NATIVE_INT, mspace, dataspace);
        }
        else {
            /* This is used for METIS to determine how many nodes an element must shared to be
             * considered as neighbors. */
            num_nodes_per_face = 1;
            cout << "The mesh file does not have a " << DSET_NNODE_PER_FACE << " dataset. "
                 << "Please recreate it to avoid this deprecated usage." << endl;
        }

        // resize internal structures
        eptr.resize(num_elems + 1);
        eind.resize(num_elems * num_nodes_per_elem); // only valid for one element group
        coord.resize(num_nodes);
        IF_to_elem.resize(nIF);
        nIF_in_elem.resize(num_elems);
        elem_to_IF.resize(num_elems);
        nBG_in_elem.resize(num_elems);
        elem_to_BF.resize(num_elems);
        for (unsigned i = 0; i < num_elems; i++) {
            nIF_in_elem[i] = 0.0;
            elem_to_IF[i].resize(6); // max number of faces
            nBG_in_elem[i] = 0.0;
            elem_to_BF[i].resize(6); // max number of faces
        }
        // resize partition ID container
        elem_partition.resize(num_elems);
        node_partition.resize(num_nodes);

        // fetch elemID -> nodeID
        dims[0] = num_elems;
        dims[1] = num_nodes_per_elem;
        rank = 2;
        mspace = DataSpace(rank, dims);
        dataset = file.openDataSet(DSET_ELEM_TO_NODES);
        dataspace = dataset.getSpace();
        // ordering (row-major): elemID X nodeID
        dataset.read(eind.data(), PredType::NATIVE_INT, mspace, dataspace);

        // fetch node coordinates
        vector<rtype> buff(dim * num_nodes, 0.);
        dims[0] = num_nodes;
        dims[1] = dim;
        rank = 2;
        mspace = DataSpace(rank, dims);
        dataset = file.openDataSet(DSET_NODE_COORD);
        dataspace = dataset.getSpace();
        // ordering (row-major): nodeID X coordinates
        dataset.read(buff.data(), PredType::NATIVE_DOUBLE, mspace, dataspace);
        // fill coordinates
        for (unsigned iNode = 0; iNode < num_nodes; iNode++) {
            coord[iNode].resize(dim);
            for (unsigned idim = 0; idim < dim; idim++) {
                coord[iNode][idim] = buff[iNode * dim + idim];
            }
        }

        // fetch IFace -> elem and IFace -> node
        vector<int> buff_int;
        buff_int.resize(nIF * 6);
        dims[0] = nIF;
        dims[1] = 6;
        rank = 2;
        mspace = DataSpace(rank, dims);
        dataset = file.openDataSet(DSET_IFACE_DATA);
        dataspace = dataset.getSpace();
        dataset.read(buff_int.data(), PredType::NATIVE_INT, mspace, dataspace);
        vector<unordered_set<int> > already_created(num_elems);
        for (unsigned i = 0; i < nIF; i++) {
            IF_to_elem[i].resize(6);
            IF_to_elem[i][0] = buff_int[6 * i + 0]; // left element ID
            IF_to_elem[i][1] = buff_int[6 * i + 1]; // face ID for the left element
            IF_to_elem[i][2] = buff_int[6 * i + 2]; // orientation for the left element
            IF_to_elem[i][3] = buff_int[6 * i + 3]; // right element ID
            IF_to_elem[i][4] = buff_int[6 * i + 4]; // face ID for the right element
            IF_to_elem[i][5] = buff_int[6 * i + 5]; // orientation for the right element

            // Set reverse mapping
            elem_to_IF[buff_int[6 * i + 0]][nIF_in_elem[buff_int[6 * i + 0]]] = i;
            elem_to_IF[buff_int[6 * i + 3]][nIF_in_elem[buff_int[6 * i + 3]]] = i;
            nIF_in_elem[buff_int[6 * i + 0]]++;
            nIF_in_elem[buff_int[6 * i + 3]]++;
        }

        // fill eptr that indicates where data for node i in eind is
        idx_t counter = 0;
        for (unsigned i = 0; i < num_elems + 1; i++) {
            eptr[i] = counter;
            counter += num_nodes_per_elem;
        }

        // read boundary faces
        if (BFGnames.size() > 0) {
            nBFG = BFGnames.size();
            nBF = 0;

            int ibface_global = 0;

            for (string BFG_name: BFGnames) {
                // fetch the number of faces in the current boundary face group
                string dset_name = "BFG_" + BFG_name + "_nBFace";
                dataset = file.openDataSet(dset_name);
                dataspace = dataset.getSpace();
                dims[0] = 1;
                rank = 1;
                mspace = DataSpace(rank, dims);
                int nBface_in_group = -1;
                dataset.read(&nBface_in_group, PredType::NATIVE_INT, mspace, dataspace);
                nBF += nBface_in_group;
                BFG_to_nBF[BFG_name] = nBface_in_group;
                BFG_to_data[BFG_name].resize(nBface_in_group);

                // fetch the boundary data for this boundary face group
                dims[0] = nBface_in_group;
                dims[1] = 3;
                rank = 2;
                mspace = DataSpace(rank, dims);
                dset_name = "BFG_" + BFG_name + "_BFaceData";
                dataset = file.openDataSet(dset_name);
                dataspace = dataset.getSpace();
                vector<int> buff_BC(dims[0] * dims[1], 0);
                dataset.read(buff_BC.data(), PredType::NATIVE_INT, mspace, dataspace);
                for (int i = 0; i < nBface_in_group; i++) {
                    BFG_to_data[BFG_name][i].resize(3);
                    BFG_to_data[BFG_name][i][0] = buff_BC[3 * i + 0];
                    BFG_to_data[BFG_name][i][1] = buff_BC[3 * i + 1];
                    BFG_to_data[BFG_name][i][2] = buff_BC[3 * i + 2];

                    // Set reverse mapping
                    elem_to_BF[buff_BC[3 * i + 0]][nBG_in_elem[buff_BC[3 * i + 0]]] =
                        ibface_global;
                    nBG_in_elem[buff_BC[3 * i + 0]]++;
                    ibface_global++;
                }
            }
        }
        else {
            nBFG = 0;
            nBF = 0;
        }
    }

    // catch failure caused by the H5File operations
    catch (H5::FileIException &error) {
        error.printErrorStack();
    }
    // catch failure caused by the DataSet operations
    catch (H5::DataSetIException &error) {
        error.printErrorStack();
    }
    // catch failure caused by the DataSpace operations
    catch (H5::DataSpaceIException &error) {
        error.printErrorStack();
    }
}

//inline void Mesh::partition_manually() {
//    cout << "WARNING: MANUAL PARTITION REQUESTED! DID YOU REALLY HARDCODE IT?" << endl;
//
//    int N = 32;
//    int M = 4; // subdivision in each direction
//    int A = N / M;
//    assert(num_elems == N*N*N);
//    assert(num_partitions == M*M);
//
//    for (int index = 0; index < num_elems; index++) {
//        int k = index % N;
//        int j = ((index - k) / N) % N;
//        int i = ((index - k - j*N) / N / N) % N;
//
//        int ipart = ((int) i / A) * M + (int) (j/A);
//        elem_partition[index] = ipart;
//    }
//
//    partitioned = true;
//}

inline void Mesh::partition() {
    idx_t objval;
    idx_t num_common = num_nodes_per_face;
    idx_t ne = num_elems;
    idx_t nn = num_nodes;
    // Use default METIS options
    idx_t options[METIS_NOPTIONS];
    METIS_SetDefaultOptions(options);

    // If there are multiple partitions, then run METIS
    if (num_partitions > 1) {
        /*
        Call METIS to partition the mesh.

        Inputs:
        -------
            ne - number of elements (must be an int)
            nn - number of nodes (must be an int)
            eptr - array of length num_elems pointing to index in eind that
                contains this element's node IDs
            eind - array of size (num_elems, num_nodes_per_elem) storing the
                node indices of each element
            vwgt - array of size ne with the weights for each element. For now,
                   not used.
            vsize - not completely sure what this does
            num_common - number of common nodes that two elements must have to
                make them neighbors (for example - for tets, ncommon = 3)
            num_partitions - number of partitions desired
            tpwgts - array of size num_partitions with the weights for each
                partition
            options - extra available METIS options
            objval - not completely sure what this does

        Outputs:
        --------
            elem_partition - vector of size num_elems that stores the partition
                vector for the elements
            node_partition - vector of size num_nodes that stores the partition
                vector for the nodes
        */
        // TODO: The node partition is never actually used. It should be
        // removed.
        int ierr = METIS_PartMeshDual(&ne, &nn, eptr.data(), eind.data(),
                NULL, NULL, &num_common, &num_partitions, NULL, options, &objval,
                elem_partition.data(), node_partition.data());
        if (ierr != METIS_OK) {
            throw FatalException("Error when partitioning the mesh with METIS!");
        }
    }

    // Otherwise, every element/node is on partition 0
    else {
        fill(elem_partition.begin(), elem_partition.end(), 0);
        fill(node_partition.begin(), node_partition.end(), 0);
    }

    // Count how many elements are contained within each partition
    std::vector<int> elem_partition_size(num_partitions, 0);
    for (auto it = elem_partition.begin(); it != elem_partition.end(); it++) {
        elem_partition_size[*it]++;
    }
    num_elems_part = elem_partition_size[rank];

    // TODO: Remove the node partition stuff from earlier, this is the new way
    // Count how many nodes are contained within each partition
    std::set<int> global_node_IDs;
    for (unsigned i = 0; i < num_elems; i++) {
        auto rank_i = (unsigned)elem_partition[i];
        if (rank == rank_i) {
            // Loop over nodes on this element
            for (unsigned j = 0; j < num_nodes_per_elem; j++) {
                // Get global ID of this node
                auto global_node_ID = eind[i * num_nodes_per_elem + j];
                // If this partition doesn't already contain this node
                if (not global_node_IDs.count(global_node_ID)) {
                    // Add this node to the partition
                    global_node_IDs.insert(global_node_ID);
                }
            }
        }
    }
    num_nodes_part = global_node_IDs.size();

    // Count how many interior faces are contained within each partition
    std::vector<int> iface_partition_size(num_partitions, 0);
    std::vector<int> ghost_faces_vector;
    std::vector<std::unordered_set<int> > sets_of_neighbor_ranks(num_ranks);
    for (unsigned i = 0; i < nIF; i++) {
        // Get left element ID
        auto elem_ID = IF_to_elem[i][0];
        // Get the rank
        auto left_rank = elem_partition[elem_ID];
        // Increment the size of the face partition on the left element's rank
        iface_partition_size[left_rank]++;
        // Get right element ID
        elem_ID = IF_to_elem[i][3];
        // Get the rank
        auto right_rank = elem_partition[elem_ID];
        // Check if this face is on a rank boundary
        if (left_rank != right_rank) {
            // Increment the size of the face partition on the right element's
            // rank, but ONLY if it's a different rank. No duplicates within a
            // rank, only at rank boundaries.
            iface_partition_size[right_rank]++;
            // If the left and right rank are not the same, then this face is a
            // ghost face. Add it to the ghost faces vector.
            ghost_faces_vector.push_back(i);
            // Add this face's neighbor ranks to each corresponding rank
            sets_of_neighbor_ranks[left_rank].insert(right_rank);
            sets_of_neighbor_ranks[right_rank].insert(left_rank);
        }
    }
    num_ifaces_part = iface_partition_size[rank];
    num_neighbor_ranks = sets_of_neighbor_ranks[rank].size();

    // Size views accordingly
    Kokkos::resize(local_to_global_elem_IDs, num_elems_part);
    Kokkos::resize(local_to_global_node_IDs, num_nodes_part);
    Kokkos::resize(local_to_global_iface_IDs, num_ifaces_part);
    Kokkos::resize(elem_to_node_IDs, num_elems_part, num_nodes_per_elem);
    Kokkos::resize(node_coords, num_nodes_part, dim);
    Kokkos::resize(interior_faces, num_ifaces_part, 8);
    Kokkos::resize(neighbor_ranks, num_neighbor_ranks);
    Kokkos::resize(num_faces_per_rank_boundary, num_neighbor_ranks);
    // Create host mirrors from these views
    h_local_to_global_elem_IDs = Kokkos::create_mirror_view(local_to_global_elem_IDs);
    h_local_to_global_node_IDs = Kokkos::create_mirror_view(local_to_global_node_IDs);
    h_local_to_global_iface_IDs = Kokkos::create_mirror_view(local_to_global_iface_IDs);
    h_elem_to_node_IDs = Kokkos::create_mirror_view(elem_to_node_IDs);
    h_node_coords = Kokkos::create_mirror_view(node_coords);
    h_interior_faces = Kokkos::create_mirror_view(interior_faces);
    h_neighbor_ranks = Kokkos::create_mirror_view(neighbor_ranks);
    h_num_faces_per_rank_boundary = Kokkos::create_mirror_view(num_faces_per_rank_boundary);

    // Set neighbor ranks
    unsigned counter = 0;
    for (auto rank : sets_of_neighbor_ranks[rank]) {
        h_neighbor_ranks(counter) = rank;
        counter++;
    }

    // Store the element IDs and elem_to_node_IDs on each partition
    counter = 0;
    for (unsigned i = 0; i < num_elems; i++) {
        auto rank_i = (unsigned)elem_partition[i];
        if (rank == rank_i) {
            // Mapping from local to global
            h_local_to_global_elem_IDs(counter) = i;
            // Node IDs of each element on this partition
            for (unsigned j = 0; j < num_nodes_per_elem; j++) {
                h_elem_to_node_IDs(counter, j) = eind[i * num_nodes_per_elem + j];
            }
            counter++;
        }
    }

    // Store the node IDs and node coordinates on each partition
    counter = 0;
    for (auto it = global_node_IDs.begin(); it != global_node_IDs.end(); it++) {
        // Global node ID of this node
        auto global_node_ID = *it;
        // Mapping from local to global
        h_local_to_global_node_IDs(counter) = global_node_ID;
        // Node coordinates of this node
        for (unsigned k = 0; k < dim; k++) {
            h_node_coords(counter, k) = coord[global_node_ID][k];
        }
        counter++;
    }

    // Store the interior face information on each partition
    num_gfaces_part = 0;
    counter = 0;
    for (unsigned i = 0; i < nIF; i++) {
        // Element rank on the left and right
        auto left_rank = (unsigned)elem_partition[IF_to_elem[i][0]];
        auto right_rank = (unsigned)elem_partition[IF_to_elem[i][3]];
        if (rank == left_rank or rank == right_rank) {
            // Mapping from local to global
            h_local_to_global_iface_IDs(counter) = i;
            // Set rank on left and right
            h_interior_faces(counter, 0) = left_rank;
            h_interior_faces(counter, 4) = right_rank;
            // If they're not the same, increment the ghost faces
            if (left_rank != right_rank) { num_gfaces_part++; }

            // Neighbors, reference face IDs, and orientations
            for (unsigned j = 0; j < 3; j++) {
                // On the left
                h_interior_faces(counter, j + 1) = IF_to_elem[i][j];
                // On the right
                h_interior_faces(counter, j + 5) = IF_to_elem[i][j + 3];
            }
            counter++;
        }
    }

    // Get number of faces in each rank boundary
    for (unsigned i = 0; i < ghost_faces_vector.size(); i++) {
        // Get global face ID
        auto global_face_ID = ghost_faces_vector[i];

        // Check if this ghost face exists on this partition
        unsigned face_ID = h_get_local_iface_ID(global_face_ID);
        if (face_ID != UINT_MAX) {
            // Get left and right rank
            auto rank_L = h_interior_faces(face_ID, 0);
            auto rank_R = h_interior_faces(face_ID, 4);

            // If this rank is on the left, then count this ghost face on the
            // right rank's index
            if (rank == rank_L) {
                // Get the rank neighbor index
                int index;
                for (unsigned j = 0; j < num_neighbor_ranks; j++) {
                    if (h_neighbor_ranks(j) == rank_R) {
                        index = j;
                        break;
                    }
                }
                // Increment
                h_num_faces_per_rank_boundary(index)++;
            // If this rank is on the right, then count this ghost face on the
            // left rank's index
            } else if (rank == rank_R) {
                // Get the rank neighbor index
                int index;
                for (unsigned j = 0; j < num_neighbor_ranks; j++) {
                    if (h_neighbor_ranks(j) == rank_L) {
                        index = j;
                        break;
                    }
                }
                // Increment
                h_num_faces_per_rank_boundary(index)++;
            }
        }
    }

    // Allocate ghost faces of each neighbor rank
    ghost_faces = new Kokkos::View<unsigned*>[num_neighbor_ranks];
    for (unsigned i = 0; i < num_neighbor_ranks; i++) {
        Kokkos::resize(ghost_faces[i], h_num_faces_per_rank_boundary(i));
    }

    // Set ghost faces of each neighbor rank
    vector<int> gface_counter(num_neighbor_ranks, 0);
    for (unsigned i = 0; i < ghost_faces_vector.size(); i++) {
        // Get global face ID
        auto global_face_ID = ghost_faces_vector[i];

        // Check if this ghost face exists on this partition
        unsigned face_ID = h_get_local_iface_ID(global_face_ID);
        if (face_ID != UINT_MAX) {
            // Get left and right rank
            auto rank_L = h_interior_faces(face_ID, 0);
            auto rank_R = h_interior_faces(face_ID, 4);

            // If this rank is on the left, then add this ghost face to the
            // right rank's index
            int index;
            if (rank == rank_L) {
                // Get the rank neighbor index
                for (unsigned j = 0; j < num_neighbor_ranks; j++) {
                    if (h_neighbor_ranks(j) == rank_R) {
                        index = j;
                        break;
                    }
                }
            // If this rank is on the right, then add this ghost face to the
            // left rank's index
            } else if (rank == rank_R) {
                // Get the rank neighbor index
                for (unsigned j = 0; j < num_neighbor_ranks; j++) {
                    if (h_neighbor_ranks(j) == rank_L) {
                        index = j;
                        break;
                    }
                }
            }

            // Set the ghost face - must be done in a parallel_for since
            // ghost_face is on the device
            auto ghost_faces_view = ghost_faces[index];
            auto gface_count = gface_counter[index];
            Kokkos::parallel_for(1, KOKKOS_LAMBDA(const int& i) {
                    ghost_faces_view(gface_count) = global_face_ID;
            });
            gface_counter[index]++;
        }
    }

    cout << "Ghost faces of rank " << rank << ":" << endl;
    for (unsigned i = 0; i < num_neighbor_ranks; i++) {
        auto ghost_faces_view = ghost_faces[i];
        Kokkos::parallel_for(h_num_faces_per_rank_boundary(i),
                KOKKOS_LAMBDA(const int& j) {
                // printf("%u\n", ghost_faces_view(j));
        });
        cout << endl;
    }
    cout << endl;

    cout << "CHECK VIEW" << endl;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 8; j++) {
            cout << h_interior_faces(i, j) << "  ";
        }
        cout << endl;
    }
    cout << "END CHECK VIEW" << endl;

    // Print
    for (unsigned rank_i = 0; rank_i < num_ranks; rank_i++) {
        if (rank == rank_i) {
            cout << "Rank " << rank << " has elements:" << endl;
            for (unsigned i = 0; i < num_elems_part; i++) {
                cout << h_local_to_global_elem_IDs(i) << endl;
            }
            cout << "Rank " << rank << " has nodes:" << endl;
            for (unsigned i = 0; i < num_nodes_part; i++) {
                cout << h_local_to_global_node_IDs(i) << endl;
            }
            cout << "Rank " << rank << " has interior faces:" << endl;
            for (unsigned i = 0; i < num_ifaces_part; i++) {
                cout << h_local_to_global_iface_IDs(i) << endl;
            }
        }
    }

    // Deallocate global mesh data
    vector<int>().swap(eind);
    vector<int>().swap(eptr);
    vector<int>().swap(elem_partition);
    vector<int>().swap(node_partition);
    vector<vector<rtype>>().swap(coord);
    vector<vector<int>>().swap(IF_to_elem);

    // Copy from host to device
    copy_from_host_to_device();

    partitioned = true;
}

inline void Mesh::copy_from_host_to_device() {
    // Use deep_copy for Views
    Kokkos::deep_copy(num_faces_per_rank_boundary, h_num_faces_per_rank_boundary);
    Kokkos::deep_copy(neighbor_ranks,              h_neighbor_ranks);
    Kokkos::deep_copy(local_to_global_node_IDs,    h_local_to_global_node_IDs);
    Kokkos::deep_copy(local_to_global_elem_IDs,    h_local_to_global_elem_IDs);
    Kokkos::deep_copy(local_to_global_iface_IDs,   h_local_to_global_iface_IDs);
    Kokkos::deep_copy(node_coords,                 h_node_coords);
    Kokkos::deep_copy(elem_to_node_IDs,            h_elem_to_node_IDs);
    Kokkos::deep_copy(interior_faces,              h_interior_faces);
}

inline void Mesh::copy_from_device_to_host() {
    // Use deep_copy for Views
    Kokkos::deep_copy(h_num_faces_per_rank_boundary, num_faces_per_rank_boundary);
    Kokkos::deep_copy(h_neighbor_ranks,              neighbor_ranks);
    Kokkos::deep_copy(h_local_to_global_node_IDs,    local_to_global_node_IDs);
    Kokkos::deep_copy(h_local_to_global_elem_IDs,    local_to_global_elem_IDs);
    Kokkos::deep_copy(h_local_to_global_iface_IDs,   local_to_global_iface_IDs);
    Kokkos::deep_copy(h_node_coords,                 node_coords);
    Kokkos::deep_copy(h_elem_to_node_IDs,            elem_to_node_IDs);
    Kokkos::deep_copy(h_interior_faces,              interior_faces);
}

template <class T>
KOKKOS_INLINE_FUNCTION
unsigned Mesh::search_for_local_ID(unsigned global_ID, T local_to_global_IDs) const {
    // Loop over the mapping
    unsigned local_ID = UINT_MAX;
    for (unsigned i = 0; i < local_to_global_IDs.extent(0); i++) {
        if (local_to_global_IDs[i] == global_ID) {
            local_ID = i;
            break;
        }
    }
    return local_ID;
}

inline unsigned Mesh::h_get_global_node_ID(unsigned local_node_ID) const {
    return h_local_to_global_node_IDs[local_node_ID];
}

inline unsigned Mesh::h_get_global_elem_ID(unsigned local_elem_ID) const {
    return h_local_to_global_elem_IDs[local_elem_ID];
}

inline unsigned Mesh::h_get_global_iface_ID(unsigned local_iface_ID) const {
    return h_local_to_global_iface_IDs[local_iface_ID];
}

inline unsigned Mesh::h_get_local_node_ID(unsigned global_node_ID) const {
    return search_for_local_ID<Kokkos::View<unsigned*>::HostMirror>(global_node_ID, h_local_to_global_node_IDs);
}

inline unsigned Mesh::h_get_local_elem_ID(unsigned global_elem_ID) const {
    return search_for_local_ID<Kokkos::View<unsigned*>::HostMirror>(global_elem_ID, h_local_to_global_elem_IDs);
}

inline unsigned Mesh::h_get_local_iface_ID(unsigned global_iface_ID) const {
    return search_for_local_ID<Kokkos::View<unsigned*>::HostMirror>(global_iface_ID, h_local_to_global_iface_IDs);
}

KOKKOS_INLINE_FUNCTION
unsigned Mesh::get_global_node_ID(unsigned local_node_ID) const {
    return local_to_global_node_IDs[local_node_ID];
}

KOKKOS_INLINE_FUNCTION
unsigned Mesh::get_global_elem_ID(unsigned local_elem_ID) const {
    return local_to_global_elem_IDs[local_elem_ID];
}

KOKKOS_INLINE_FUNCTION
unsigned Mesh::get_global_iface_ID(unsigned local_iface_ID) const {
    return local_to_global_iface_IDs[local_iface_ID];
}

KOKKOS_INLINE_FUNCTION
unsigned Mesh::get_local_node_ID(unsigned global_node_ID) const {
    return search_for_local_ID<Kokkos::View<unsigned*>>(global_node_ID, local_to_global_node_IDs);
}

KOKKOS_INLINE_FUNCTION
unsigned Mesh::get_local_elem_ID(unsigned global_elem_ID) const {
    return search_for_local_ID<Kokkos::View<unsigned*>>(global_elem_ID, local_to_global_elem_IDs);
}

KOKKOS_INLINE_FUNCTION
unsigned Mesh::get_local_iface_ID(unsigned global_iface_ID) const {
    return search_for_local_ID<Kokkos::View<unsigned*>>(global_iface_ID, local_to_global_iface_IDs);
}

inline string Mesh::report() const {
    stringstream msg;

    msg << string(80, '=') << endl;
    msg << "---> Mesh object reporting" << endl;
    msg << "Number of elements = " << num_elems << endl
        << "Number of nodes = " << num_nodes << endl
        << "Number of interior faces = " << nIF << endl
        << "Geometric order = " << order << endl;
    if (nBFG > 0) {
        msg << "--> Mesh has boundaries" << endl;
        msg << "Number of boundary face groups = " << nBFG << ", Number of boundary faces = " << nBF << endl;
        msg << "Groups: ";
        for (string name: BFGnames) {
            msg << name << " ";
        }
        msg << endl;
    }
    else {
        msg << "--> No boundary group. Assuming everything is periodic." << endl;
    }
    if (partitioned) {
        msg << "--> Mesh is partitioned: " << endl;
        for (int part = 0; part < num_partitions; part++) {
            msg << "Elements in partition " << rank << ": " << num_elems_part << endl;
        }
    }
    else {
        msg << "--> Mesh is not partitioned." << endl;
    }
    msg << string(80, '=') << endl << endl;

    return msg.str();
}
