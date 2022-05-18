# Installing Dependencies

These are some useful tips for installing various submodules that are used with the DG-MPI solver. It is important to note that the scripts located in [`src/other`](https://github.com/IhmeGroup/DG-MPI/tree/main/src/others) handle these dependencies implicitly and should be the primary tool for installation. 

The goal of the external builds is to have a:
1. Streamlined compilation of this code to make it user-friendly
2. Rigorously track the versions of dependencies used for future regression
   tests

The compilation of submodules is automated (via the scripts described above). Here is a list of the extra dependencies:

 - TOML11 (markup language for reading input files): no compilation needed,
   since this is header-only.
 - METIS (graph partitioning library): This is automated with a Bash script.
    ```
    ./download_dependencies.sh
    ```
- MPICH (MPI library): MPI uses local builds (not packaged with dg-mpi). The cmake file searches for the environment variable `MPI_DIR` and if not found it expects `mpicc` and `mpicxx` to be in the user's `$PATH`.
   - Ubuntu users, use `sudo apt install mpich`
   - macOS users, use `brew install mpich`. Note: Make sure you have the right compilers prior to this. Use `brew install llvm` to get the correct compilers.
   - Yellowstone: use `module load mpich`

 - HDF5 (high-performance data formatting library): This is automated with a Bash script.
    ```
    ./download_dependencies.sh
    ```
 - Kokkos:
    ```
      cd kokkos
      mkdir build
      cd build
      cmake -DCMAKE_INSTALL_PREFIX=./install -DKokkos_ENABLE_OPENMP=ON ..
      make install
    ```
    - macOSX Notes:
         For openMP support I needed to get llvm via homebrew (`brew install llvm`). I also needed to update my `.zshrc_profile` with the following:
         ```
         export PATH="/usr/local/opt/llvm/bin:$PATH"
         export CC=/usr/local/opt/llvm/bin/clang
         export CXX=/usr/local/opt/llvm/bin/clang++
         export LDFLAGS=-L/usr/local/opt/llvm/lib
         export CPPFLAGS=-I/usr/local/opt/llvm/include
         ```
         This would be similar for `.bashrc` files
    - Yellowstone Notes:
         Kokkos requires CMake >= 3.16. Yellowstone has version 3.15.4. Therefore, we need to install a local build of CMake. Use the following commands to accomplish this:

         - Download the Unix/Linux source cmake file (I used version 3.22.2) `https://cmake.org/download/`
         - Send this to yellowstone using `rsync` or whatever your favorite file transfer tool is.
         - Once you have the `*tar.gz` file where you want the CMake build to happen use these commands:
         ```
         tar -zxvf cmake-3.22.2.tar.gz
         cd cmake-3.22.2
         mkdir install
         cmake -DCMAKE_USE_OPENSSL=OFF -DCMAKE_INSTALL_PREFIX=<path_to_your_cmake>/install .
         make
         make install
         ```
         - Then when you go to use CMake with Kokkos you can either give the direct path to the new cmake command when executing the cmake command for Kokkos above or change your default CMake to the newly installed version (by changing the paths / unloading the module for CMake).

 - Kokkos-kernels:
    ```
      cd kokkos-kernels
      mkdir build
      cd build
      cmake -DCMAKE_INSTALL_PREFIX=./install -DKokkos_ROOT=../../kokkos/build/install ..
      make install
    ```
    - Yellowstone / macOS notes : See Kokkos build description above.
