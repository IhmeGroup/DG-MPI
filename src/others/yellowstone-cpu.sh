#!/bin/bash
# Yellowstone specific build instructions (CPU version) - updated as of 2/14/22

# This file should be called from the 'DG-MPI/src/others' folder.

# NOTE: Make sure you follow directions to install cmake version 3.22 on the
# github page (see externals/README.md) and then run this command as follows:

# cmake=<PATH_TO_CMAKE>/cmake-3.22.2/bin/cmake ./yellowstone-cpu.sh

# Load necessary modules
module purge
module load gnu8/8.3.0
#module load hdf5/1.10.5
module load mpich/3.3.1

set -e

# move to the home dir
cd ../../

git submodule update --init

mkdir -p build_cpu/build_externals
cd build_cpu/build_externals
mkdir -p metis
mkdir -p kokkos/build/install
mkdir -p kokkos-kernels/build/install
cd ../../

cd externals
build_path=../../../../build_cpu/build_externals/metis/install ./download_dependencies.sh

echo "========================"
echo "    METIS BUILD DONE    "
echo "========================"

cd ../build_cpu/build_externals/kokkos/build

$cmake -DCMAKE_INSTALL_PREFIX=./install \
    -DKokkos_ENABLE_OPENMP=ON \
    -DKokkos_CXX_STANDARD=17 \
    ../../../../externals/kokkos
make -j install

echo "========================"
echo "   KOKKOS BUILD DONE    "
echo "========================"

cd ../../kokkos-kernels/build

$cmake -DCMAKE_INSTALL_PREFIX=./install -DKokkos_ROOT=../../kokkos/build/install \
    ../../../../externals/kokkos-kernels
make -j install

echo "================================"
echo "   KOKKOS-KERNELS BUILD DONE    "
echo "================================"


cd ../../../

$cmake -DGPU_BUILD=0 ..
make -j"${nthreads}"

echo "========================"
echo "   DG-MPI BUILD DONE    "
echo "========================"
