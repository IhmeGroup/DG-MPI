#include <algorithm>
#include <iostream>
#include <map>
#include <numeric>
#include <string>
#include "H5Cpp.h"
// #include "hdf5.h"
#include "metis.h"
#include "toml11/toml.hpp"
#include "common/my_exceptions.h"
#include "mesh/mesh.h"

using namespace std;
using H5::PredType, H5::DataSpace, H5::DataSet;

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

Mesh::Mesh(const toml::value &input_info, MemoryNetwork& network)
    : network{network} {
    auto mesh_info = toml::find(input_info, "Mesh");
    string mesh_file_name = toml::find<string>(mesh_info, "file");
    num_partitions = toml::find<int>(mesh_info, "npartitions");

    // TODO figure out boundaries from the HDF5 file directly (Kihiro 2021/03/04)
    if (input_info.contains("Boundaries")) {
        BFGnames = toml::find<vector<string>>(input_info, "Boundaries", "names");
    }
    else {
        nBFG = 0;
        nBF = 0;
    }
    read_mesh(mesh_file_name);
}

void Mesh::read_mesh(const string &mesh_file_name) {
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
        dataset.read(&nnode_per_elem, PredType::NATIVE_INT, mspace, dataspace);
        // number of nodes per faces
        if (H5Lexists(file.getId(), DSET_NNODE_PER_FACE.c_str(), H5P_DEFAULT)) {
            dataset = file.openDataSet(DSET_NNODE_PER_FACE);
            dataspace = dataset.getSpace();
            dataset.read(&nnode_per_face, PredType::NATIVE_INT, mspace, dataspace);
        }
        else {
            /* This is used for METIS to determine how many nodes an element must shared to be
             * considered as neighbors. */
            nnode_per_face = 1;
            cout << "The mesh file does not have a " << DSET_NNODE_PER_FACE << " dataset. "
                 << "Please recreate it to avoid this deprecated usage." << endl;
        }

        // resize internal structures
        eptr.resize(num_elems + 1);
        eind.resize(num_elems * nnode_per_elem); // only valid for one element group
        coord.resize(num_nodes);
        IF_to_elem.resize(nIF);
        nIF_in_elem.resize(num_elems);
        elem_to_IF.resize(num_elems);
        nBG_in_elem.resize(num_elems);
        elem_to_BF.resize(num_elems);
        for (int i = 0; i < num_elems; i++) {
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
        dims[1] = nnode_per_elem;
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
        for (int iNode = 0; iNode < num_nodes; iNode++) {
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
        for (int i = 0; i <  + 1; i++) {
            eptr[i] = counter;
            counter += nnode_per_elem;
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

    // partition the mesh using METIS
    partition();
}

void Mesh::partition_manually() {
    cout << "WARNING: MANUAL PARTITION REQUESTED! DID YOU REALLY HARDCODE IT?" << endl;

    int N = 32;
    int M = 4; // subdivision in each direction
    int A = N / M;
    int M2 = 2;
    int A2 = N / (M*M2);
    assert(num_elems == N*N*N);
    assert(num_partitions == M*M);

    for (int index = 0; index < num_elems; index++) {
        int k = index % N;
        int j = ((index - k) / N) % N;
        int i = ((index - k - j*N) / N / N) % N;

        int ipart = ((int) i / A) * M + (int) (j/A);
        elem_partition[index] = ipart;
    }

    partitioned = true;
}

void Mesh::partition() {
    idx_t objval;
    idx_t num_common = nnode_per_face;

    // If there are multiple partitions, then run METIS
    if (num_partitions > 1) {
        /*
        Call METIS to partition the mesh.

        Inputs:
        -------
            num_elems - number of elements
            num_nodes - number of nodes
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
        int ierr = METIS_PartMeshDual(&num_elems, &num_nodes, eptr.data(), eind.data(),
                NULL, NULL, &num_common, &num_partitions, NULL, NULL, &objval,
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

    partitioned = true;
}

string Mesh::report() const {
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
            long int N = count(elem_partition.begin(), elem_partition.end(), part);
            msg << "Elements in partition " << part << ": " << N << endl;
        }
    }
    else {
        msg << "--> Mesh is not partitioned." << endl;
    }
    msg << string(80, '=') << endl << endl;

    return msg.str();
}
