#ifndef DG_MESH_H
#define DG_MESH_H

#include <map>
#include <string>
#include <vector>
#include <Kokkos_Core.hpp>
#include <Kokkos_UnorderedMap.hpp>
#include "toml11/toml.hpp"
#include "common/defines.h"

// Forward declaration
class MemoryNetwork;


/*! \brief Mesh class
 *
 * This class is really just an intermediate representation of the mesh.
 * The Gmsh file gets processed into HDF5 using Eric's python tool.
 * Then this class reads the HDF5 file.
 * Then the data is transfered to a region.
 *
 * We don't read it into a region directly because when we first wrote the code,
 * it wasn't clear how to format the HDF5 file to interface directly with Legion.
 */
class Mesh {
    public:
        /*
        Constructor for the mesh.

        Inputs:
        -------
        input_info - TOML input file
        network - memory network object
        mesh_file_name - optional name of mesh file. If not specified, looks for
            the name in the input file.
        */
        Mesh(const toml::value& input_info, const MemoryNetwork& network,
                std::string mesh_file_name = "");
        ~Mesh();

        /*! \brief Read the HDF5 mesh file directly
         *
         * Currently allowed here for testing.
         * Will probably not work if there is any boundary.
         *
         * @param mesh_file_name
         */
        void read_mesh(const std::string &mesh_file_name);

        // Copy data in host mirrors to the device Views.
        void copy_from_host_to_device();

        // Copy data in device Views to host mirrors.
        void copy_from_device_to_host();

        /*! \brief Report mesh object
         *
         * @return
         */
        std::string report() const;

    private:
        /*! \brief Partition the mesh using METIS
         *
         */
        void partition();

        /*! \brief Custom manual partitionning method
         *
         */
        //void partition_manually();

    public:
        unsigned dim; //!< number of spatial dimentions
        unsigned order; //!< geometric order
        unsigned num_elems; //!< number of elements
        unsigned num_nodes; //!< number of nodes
        unsigned nIF; //!< number of interior faces
        unsigned nBFG; //!< number of boundary face groups (BFG)
        unsigned nBF; //!< total number of boundary faces
        unsigned num_nodes_per_elem; //!< number of nodes per element
        unsigned num_nodes_per_face; //!< number of nodes per face
        int num_partitions = 1; //!< number of partitions requested
        std::vector<std::string> BFGnames; //!< name of boundary groups
        std::map<std::string, int> BFG_to_nBF; //!< map from BFG name to number of boundary faces in that group
        /*! \brief Map from BFG name to boundary data
         *
         * BFG[name][iBF] = vector of size 3.
         *
         * Each element of the vector contains the following information;
         * - the element ID
         * - the face ID from the point of view of the element
         * - the face orientation from the point of view of the element
         */
        std::map<std::string, std::vector<std::vector<int>>> BFG_to_data;
        /*! \brief Vector containing node coordinates for each node
         *
         * coord[inode] = vector of size number of spatial dimensions.
         */
        std::vector<std::vector<rtype>> coord;
        /*! \brief Vector relating faces to adjacent elements
         *
         * IF_to_elem[iIF] = vector of size 3*2=6.
         *
         * Each element of the vector contains the following information
         * (first for left then for right element):
         * - element ID
         * - face ID from the point of view of the element
         * - face orientation from the point of view of the element
         */
        std::vector<std::vector<int>> IF_to_elem;
        std::vector<int> nIF_in_elem;
        std::vector<std::vector<int>> elem_to_IF;
        std::vector<int> nBG_in_elem;
        std::vector<std::vector<int>> elem_to_BF;
        const MemoryNetwork& network;
    public:
        std::vector<int> eptr; // for metis
        std::vector<int> eind; // for metis
        std::vector<int> elem_partition; // vector of partition ID for each element
        std::vector<int> node_partition; // vector of partition ID for each node
        std::vector<int> iface_partition; // vector of partition ID for each interior face
    public:
        // This is the stuff that should persist throughout the simulation
        // (previous things, especially std::vector's, are only used
        // for the mesh reading step). All data structures above this are
        // subject to further code review/removal - treat the below as the true
        // mesh API for now.

        // Number of elements in this partition
        unsigned num_elems_part;
        // Number of nodes in this partition
        unsigned num_nodes_part;
        // Number of interior faces in this partition. This includes both:
        //  - local interior faces (faces where rank_L == rank_R)
        //  - ghost faces (faces where rank_L != rank_R)
        // This means that ghost faces are copied (one on each neighbor rank).
        unsigned num_ifaces_part;
        // Number of ghost faces in this partition
        unsigned num_gfaces_part;
        // Number of ranks that neighbor this rank
        unsigned num_neighbor_ranks;
        // View containing the number of faces shared between this rank and each
        // of its rank boundaries
        Kokkos::View<unsigned*> num_faces_per_rank_boundary;
        // View containing all the ranks that neighbor this rank
        Kokkos::View<unsigned*> neighbor_ranks;
        /* For below, 'local' refers to partition-local, wheres 'global'
         * refers to the value for the global mesh. It does not have anything to
         * do with reference vs. physical space. */
        // Views that set a mapping from local to global node IDs, element IDs,
        // and interior face IDs
        Kokkos::View<unsigned*> local_to_global_node_IDs;
        Kokkos::View<unsigned*> local_to_global_elem_IDs;
        Kokkos::View<unsigned*> local_to_global_iface_IDs;
        // Maps that set a mapping from global to local node IDs, element IDs,
        // and interior face IDs
        Kokkos::UnorderedMap<unsigned, unsigned> global_to_local_node_IDs;
        Kokkos::UnorderedMap<unsigned, unsigned> global_to_local_elem_IDs;
        Kokkos::UnorderedMap<unsigned, unsigned> global_to_local_iface_IDs;
        // Jagged array containing the global face IDs of the ghost faces. The
        // first index represents the neighboring rank index, and the second
        // index represents the ghost face index in that neighboring rank.
        /* Example:
         * if neighbor_ranks = {2, 4, 9},
         * and num_faces_per_rank_boundary = {3, 2, 4}, then ghost_faces looks
         * like:
         * ghost_faces = {
         *     {*, *, *},
         *     {*, *},
         *     {*, *, *, *}
         * }
         * where the *'s represent global face IDs. The first row (with 3 faces)
         * are those neighboring rank 2, the second row (with 2 faces) are those
         * neighboring rank 4, and the third row (with 4 faces) are those
         * neighboring rank 9. */
        unsigned** ghost_faces;
        // View containing the coordinates for each node on this partition
        Kokkos::View<rtype**> node_coords;
        // View containing the mapping from elements to global node IDs on this
        // partition
        Kokkos::View<unsigned**> elem_to_node_IDs;
        // View containing the interior face information, of shape
        // [num_ifaces_part, 8], where the 8 pieces of data are:
        // the rank, element ID, ref. face ID, and orientation on the left, and
        // the rank, element ID, ref. face ID, and orientation on the right.
        Kokkos::View<unsigned**> interior_faces;

        // Everything below is the host mirror version of the data structures
        // above. The same comments from above apply here.
        Kokkos::View<unsigned*>::HostMirror h_num_faces_per_rank_boundary;
        Kokkos::View<unsigned*>::HostMirror h_neighbor_ranks;
        Kokkos::View<unsigned*>::HostMirror h_local_to_global_node_IDs;
        Kokkos::View<unsigned*>::HostMirror h_local_to_global_elem_IDs;
        Kokkos::View<unsigned*>::HostMirror h_local_to_global_iface_IDs;
        Kokkos::UnorderedMap<unsigned, unsigned>::HostMirror h_global_to_local_node_IDs;
        Kokkos::UnorderedMap<unsigned, unsigned>::HostMirror h_global_to_local_elem_IDs;
        Kokkos::UnorderedMap<unsigned, unsigned>::HostMirror h_global_to_local_iface_IDs;
        Kokkos::View<rtype**>::HostMirror h_node_coords;
        Kokkos::View<unsigned**>::HostMirror h_elem_to_node_IDs;
        Kokkos::View<unsigned**>::HostMirror h_interior_faces;

        // geometric basis is a member of the mesh
        Basis::Basis gbasis;
    private:
        bool partitioned = false; //!< boolean indicating whether the mesh is partitioned
};

#endif //DG_MESH_H
