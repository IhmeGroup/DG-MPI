The following set of scripts can be used for the build process. For each of these it is assumed that you have your own build of cmake since Kokkos requires a cmake version newer than the one provided on the Yellowstone cluster.

For a CPU build (yellowstone-cpu.sh):
This file should be called from the 'DG-MPI/src/others' folder.

NOTE: Make sure you follow directions to install cmake version 3.22 on the
github page (see externals/README.md) and then run this command as follows:

`cmake=<PATH_TO_CMAKE>/cmake-3.22.2/bin/cmake ./yellowstone-cpu.sh`
