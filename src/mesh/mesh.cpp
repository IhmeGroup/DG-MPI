//
// Created by kihiro on 1/23/20.
//

#include <algorithm>
#include <iostream>
#include <map>
#include <numeric>
#include <string>
#include "H5Cpp.h"
#include "hdf5.h"
#include "metis.h"
#include "toml11/toml.hpp"
#include "common/my_exceptions.h"
#include "mesh/mesh.h"

using namespace std;
using namespace H5;

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

Mesh::Mesh(const toml::value &input_info) : partitioned(false) {
    auto mesh_info = toml::find(input_info, "Mesh");
    string mesh_file_name = toml::find<string>(mesh_info, "file");
    npart = toml::find<int>(mesh_info, "npartitions");
    npart_lvl2 = toml::find_or<int>(mesh_info, "npartitions_lvl2", -1);

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
        H5File file(mesh_file_name, H5F_ACC_RDONLY);
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
        dataset.read(&nelem, PredType::NATIVE_INT, mspace, dataspace);
        // number of nodes
        dataset = file.openDataSet(DSET_NNODE);
        dataspace = dataset.getSpace();
        dataset.read(&nnode, PredType::NATIVE_INT, mspace, dataspace);
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
        eptr.resize(nelem + 1);
        eind.resize(nelem * nnode_per_elem); // only valid for one element group
        coord.resize(nnode);
        IF_to_elem.resize(nIF);
        nIF_in_elem.resize(nelem);
        elem_to_IF.resize(nelem);
        nBG_in_elem.resize(nelem);
        elem_to_BF.resize(nelem);
        for (int i = 0; i < nelem; i++) {
            nIF_in_elem[i] = 0.0;
            elem_to_IF[i].resize(6); // max number of faces
            nBG_in_elem[i] = 0.0;
            elem_to_BF[i].resize(6); // max number of faces
        }
        // resize partition ID container
        elem_part_id.resize(nelem);
        node_part_id.resize(nnode);

        // fetch elemID -> nodeID
        dims[0] = nelem;
        dims[1] = nnode_per_elem;
        rank = 2;
        mspace = DataSpace(rank, dims);
        dataset = file.openDataSet(DSET_ELEM_TO_NODES);
        dataspace = dataset.getSpace();
        // ordering (row-major): elemID X nodeID
        dataset.read(eind.data(), PredType::NATIVE_INT, mspace, dataspace);

        // fetch node coordinates
        vector<rtype> buff(dim * nnode, 0.);
        dims[0] = nnode;
        dims[1] = dim;
        rank = 2;
        mspace = DataSpace(rank, dims);
        dataset = file.openDataSet(DSET_NODE_COORD);
        dataspace = dataset.getSpace();
        // ordering (row-major): nodeID X coordinates
        dataset.read(buff.data(), PredType::NATIVE_DOUBLE, mspace, dataspace);
        // fill coordinates
        for (int iNode = 0; iNode < nnode; iNode++) {
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
        for (int i = 0; i < nelem + 1; i++) {
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
    catch (FileIException &error) {
        error.printErrorStack();
    }
    // catch failure caused by the DataSet operations
    catch (DataSetIException &error) {
        error.printErrorStack();
    }
    // catch failure caused by the DataSpace operations
    catch (DataSpaceIException &error) {
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
    assert(nelem==N*N*N);
    assert(npart==M*M);
    assert(npart_lvl2==M2*M2);

    elem_subreg_id1.resize(nelem);
    elem_subreg_id2.resize(nelem);

    for (int index=0; index<nelem; index++) {
        int k = index % N;
        int j = ((index - k) / N) % N;
        int i = ((index - k - j*N) / N / N) % N;

        int ipart = ((int) i / A) * M + (int) (j/A);
        elem_part_id[index] = ipart;
        elem_subreg_id1[index] = ipart;

        int ipart2 = ((int) i/A2) * M2 + (int) (j/A2);
        elem_subreg_id2[index] = ipart2 % (M2*M2);
    }

    partitioned = true;
}

void Mesh::partition() {
    idx_t objval;
    idx_t ncommon = nnode_per_face;
    idx_t nPart_lvl1 = npart;

    if (nPart_lvl1 > 1) {
        int ierr = METIS_PartMeshDual(&nelem, &nnode,
            eptr.data(), eind.data(),
            NULL, NULL, &ncommon, &nPart_lvl1, NULL, NULL, &objval,
            elem_part_id.data(), node_part_id.data());
        if (ierr != METIS_OK) {
            throw FatalException("Error when creating the first level partition with METIS!");
        }
    }
    else {
        fill(elem_part_id.begin(), elem_part_id.end(), 0);
    }

    // second-level partition is not supported at the moment
    if (npart_lvl2>0) {
        elem_subreg_id1.resize(nelem);
        elem_subreg_id2.resize(nelem);

        copy(elem_part_id.begin(), elem_part_id.end(), elem_subreg_id1.begin());
        fill(elem_subreg_id2.begin(), elem_subreg_id2.end(), -1);

        for (int i=0; i<npart; i++) {
            vector<int> elem_ids_in_subreg;
            map<int, int> glob_to_loc_node_id;
            map<int, int> loc_to_glob_elem_id;

            auto iter = elem_part_id.begin();
            int elem_id = 0;
            int node_id = 0;

            /* We collect every element and node that is in lvl1 sub-region i and assign a lvl1
             * sub-region local numbering.
             */
            while (iter!=elem_part_id.end()) {
                iter = find(iter, elem_part_id.end(), i);
                if (iter!=elem_part_id.end()) {
                    int ielem_glob = distance(elem_part_id.begin(), iter);
                    elem_ids_in_subreg.push_back(ielem_glob);
                    loc_to_glob_elem_id.insert({elem_id, ielem_glob});
                    elem_id++;

                    int start_glob = eptr[ielem_glob];
                    for (unsigned inode=0; inode<nnode_per_elem; inode++) {
                        int node_id_glob = eind[start_glob + inode];
                        // if it is the first time we encounter this node, add it
                        if (glob_to_loc_node_id.find(node_id_glob)==glob_to_loc_node_id.end()) {
                            glob_to_loc_node_id.insert({eind[start_glob + inode], node_id});
                            node_id ++;
                        }
                    }

                    iter++;
                }
            }

            /* We recreate the eptr and eind variables (see METIS documentation) locally in the
             * current lvl1 sub-region i.
             */
            int nElem_in_subreg = elem_ids_in_subreg.size();
            assert(elem_ids_in_subreg.size()==loc_to_glob_elem_id.size());
            int nNode_in_subreg = glob_to_loc_node_id.size();
            vector<int> eptr_in_subreg(nElem_in_subreg+1);
            vector<int> eind_in_subreg(nElem_in_subreg * nnode_per_elem);
            int start = 0;
            for (int ielem=0; ielem<nElem_in_subreg; ielem++) {
                int ielem_glob = elem_ids_in_subreg[ielem];
                int start_glob = eptr[ielem_glob];
                eptr_in_subreg[ielem] = start;
                for (unsigned inode=0; inode<nnode_per_elem; inode++) {
                    int node_id_glob = eind[start_glob + inode];
                    eind_in_subreg[start+inode] = glob_to_loc_node_id.at(node_id_glob);
                }
                start += nnode_per_elem;
            }
            eptr_in_subreg[nElem_in_subreg] = start;

            /* We are now ready to call METIS one more time on the lvl1 sub-region i.
             * Note that lvl2 sub-region ID will be in {0,...,npart_lvl2-1}.
             */
            vector<int> elem_buff(nElem_in_subreg);
            vector<int> node_buff(nNode_in_subreg);
            int nPart2 = npart_lvl2;
            int ierr =  METIS_PartMeshDual(&nElem_in_subreg, &nNode_in_subreg,
                eptr_in_subreg.data(), eind_in_subreg.data(),
                NULL, NULL, &ncommon, &nPart2, NULL, NULL, &objval,
                elem_buff.data(), node_buff.data());
            if (ierr!=METIS_OK) {
                throw FatalException("Error when creating the second level partition with METIS!");
            }

            // finally, we re-map it to the global numbering
            for (int ielem=0; ielem<nElem_in_subreg; ielem++) {
                elem_subreg_id2[loc_to_glob_elem_id.at(ielem)] = elem_buff[ielem];
            }
        }
    } // if npart_lvl2 > 0

    partitioned = true;
}

string Mesh::report() const {
    stringstream msg;

    msg << string(80, '=') << endl;
    msg << "---> Mesh object reporting" << endl;
    msg << "nElem  = " << nelem << endl
        << "nNode  = " << nnode << endl
        << "nIface = " << nIF << endl
        << "order  = " << order << endl;
    if (nBFG > 0) {
        msg << "--> Mesh has boundaries" << endl;
        msg << "nBFG   = " << nBFG << ", nBFace = " << nBF << endl;
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
        for (int part = 0; part < npart; part++) {
            long int N = count(elem_part_id.begin(), elem_part_id.end(), part);
            msg << "Elements in piece " << part << ": " << N << endl;
        }

        // second-level partition is not supported at the moment
        if (npart_lvl2 > 1) {
            vector<unsigned> range((unsigned) nelem);
            iota(range.begin(), range.end(), 0);
            msg << endl << "Second level sub-regions:" << endl;
            for (int ilvl1 = 0; ilvl1 < npart; ilvl1++) {
                for (int ilvl2 = 0; ilvl2 < npart_lvl2; ilvl2++) {
                    long int N = count_if(range.begin(), range.end(),
                        [this, ilvl1, ilvl2](unsigned int i) {
                            return elem_subreg_id1[i] == ilvl1 && elem_subreg_id2[i] == ilvl2;
                        });
                    msg << "Elements in partition (" << ilvl1 << "," << ilvl2 << "): " << N << endl;
                }
            }
        }
    }
    else {
        msg << "--> Mesh is not partitioned." << endl;
    }
    msg << string(80, '=') << endl << endl;

    return msg.str();
}
