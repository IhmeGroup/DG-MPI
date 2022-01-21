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

 - TOML11 (markup language for reading input files): no compilation needed,
   since this is header-only.
 - METIS (graph partitioning library): This is automated with a Bash script.
    ```
    ./download_dependencies.sh
    ```
 - HDF5 (high-performance data formatting library):
    ```
    cd hdf5
    mkdir build
    cd build
    cmake -DCMAKE_INSTALL_PREFIX=./install -DHDF5_BUILD_CPP_LIB=1 -DBUILD_STATIC_LIBS=0 ..
    make install
    ```
 - Kokkos:
    ```
      cd kokkos
      mkdir build
      cd build
      cmake -DCMAKE_INSTALL_PREFIX=./install -DKokkos_ENABLE_OPENMP=ON ..
      make install
    ```
    NOTE: For macOSX users -> For openMP support I needed to get llvm via homebrew (`brew install llvm`). I also needed to update my `.zshrc_profile` with the following:
    ```
    export PATH="/usr/local/opt/llvm/bin:$PATH"
    export CC=/usr/local/opt/llvm/bin/clang
    export CXX=/usr/local/opt/llvm/bin/clang++
    export LDFLAGS=-L/usr/local/opt/llvm/lib
    export CPPFLAGS=-I/usr/local/opt/llvm/include
    ```
    This would be similar for `.bashrc` files
 - Kokkos-kernels:
    ```
      cd kokkos-kernels
      mkdir build
      cd build
      cmake -DCMAKE_INSTALL_PREFIX=./install -DKokkos_ROOT=../../kokkos/build/install ..
      make install
    ```
