# Installing Dependencies

I am attempting to use Git submodules for (all?) the dependencies, in an attempt
to:
1. Streamline the compilation of this code to make it user-friendly
2. Rigorously track the versions of dependencies used for future regression
   tests

First, download and initialize the submodules by running
    git submodule update --init

Ideally the compilation of submodules should be automated (probably into the
main CMake pipeline) but for now, I will document the commands I used to
compile everything. (feel free to add some -j4 to those make commands if you
want, especially for HDF5).

NOTE: The GKLib/METIS part does not seem to be linking. For now, download METIS
separately and set the METIS_ROOT and METIS_LIBRARIES manually in ccmake.

NOTE: Actually, HDF5 isn't linking right either.

 - TOML11 (markup language for reading input files): no compilation needed,
   since this is header-only.
 - GKLib (needed by METIS):
    cd gklib
    make config cc=gcc prefix=./install
    make install
 - METIS (graph partitioning library):
    cd metis
    make config cc=gcc prefix=./install gklib_path=../gklib/build/Linux-x86_64/install
    make install
 - HDF5 (high-performance data formatting library):
    cd hdf5
    mkdir build
    cd build
    cmake -DCMAKE_INSTALL_PREFIX=./build -DHDF5_BUILD_CPP_LIB=1 -DBUILD_STATIC_LIBS=0 ..
    make install

