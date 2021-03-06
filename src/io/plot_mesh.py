import h5py
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np

quad_node_order = [0, 1, 3, 2]
triangle_node_order = [0, 1, 2]


def main():
    # Name of HDF5 file
    file_name = 'test/mpi_enabled_tests/mesh/data.h5'

    # Create HDF5 file
    h5_file = h5py.File(file_name, 'r')

    # Get attributes
    num_ranks = h5_file.attrs['Number of Ranks']
    dim = h5_file.attrs['Number of Dimensions']
    num_elems = h5_file.attrs['Number of Elements']
    num_nodes = h5_file.attrs['Number of Nodes']
    num_nodes_per_elem = h5_file.attrs['Number of Nodes Per Element']

    if num_nodes_per_elem == 4:
        node_order = quad_node_order
    elif num_nodes_per_elem == 3:
        node_order = triangle_node_order
    else:
        print('Element type not supported.')

    # Allocate
    global_node_coords = np.empty((num_nodes, dim))
    global_elem_to_node_IDs = np.empty((num_elems, num_nodes_per_elem))

    # Loop over rank groups
    elem_to_node_IDs_list = []
    for i in range(num_ranks):
        # Get group
        group = h5_file[f'Rank {i}']
        # Read data
        node_coords = group['Node Coordinates'][()]
        local_to_global_node_IDs = group['Local to Global Node IDs'][()]
        elem_to_node_IDs = group['Element Global Node IDs'][()]
        local_to_global_elem_IDs = group['Local to Global Element IDs'][()]
        # Arange into global arrays
        global_node_coords[local_to_global_node_IDs] = node_coords
        global_elem_to_node_IDs[local_to_global_elem_IDs] = elem_to_node_IDs
        elem_to_node_IDs_list.append(elem_to_node_IDs)

    # Plot
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)
    fig = plt.figure(figsize=(4,4))
    # Loop over ranks
    for i in range(num_ranks):
        color = None
        label = f'Rank {i}'
        # Loop over elements
        for elem_ID in range(elem_to_node_IDs_list[i].shape[0]):
            # Coordinates
            coords = global_node_coords[
                    elem_to_node_IDs_list[i][elem_ID]][node_order]
            # Loop them so the first point shows up again at the end
            coords = np.vstack((coords, coords[0]))
            # Plot, with all elements in the same rank being the same color, and
            # the label only used once
            p = plt.plot(coords[:, 0], coords[:, 1], linewidth=1.5, color=color,
                    label=label)[0]
            color = p.get_color()
            label = None
    plt.xlabel('$x$', fontsize=20)
    plt.ylabel('$y$', fontsize=20)
    plt.tick_params(labelsize=12)
    #plt.legend(loc='upper left', fontsize = 12)
    #plt.grid(linestyle='--')
    plt.savefig('mesh.svg', bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    main()
