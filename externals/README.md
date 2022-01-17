I am attempting to use Git submodules for (all?) the dependencies, in an attempt
to:
1. Streamline the compilation of this code to make it user-friendly
2. Rigorously track the versions of dependencies used for future regression
   tests

So far there is TOML11 (DG-Legion already had this as a submodule) but I am
adding METIS to this as well, let's see how this goes.

Ideally the compilation of submodules should be automated (probably into the
main CMake pipeline) but for now, I will document the commands I used to
compile everything. (feel free to add some -j4 to those make commands if you
want, especially for HDF5).
 - GKLib (needed by METIS):
    cd gklib
    mkdir build
    make config cc=gcc prefix=./build
    make install
 - METIS (graph partitioning library):
    cd metis
    mkdir build
    make config cc=gcc prefix=./build gklib_path=../gklib/build/Linux-x86_64/build
    make install
 - HDF5 (high-performance data formatting library):
    cd hdf5
    mkdir build
    cmake -DCMAKE_INSTALL_PREFIX=./build -DHDF5_BUILD_CPP_LIB=1 ..
    make install

