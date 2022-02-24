#!/bin/bash
# Ubuntu specific build instructions (CPU version)

# This file should be called from the 'DG-MPI/src/others' folder.

# move to the home dir
cd ../../

git submodule update --init

mkdir -p build_cpu/build_externals
cd build_cpu/build_externals
mkdir metis
mkdir -p kokkos/build/install
mkdir -p kokkos-kernels/build/install
cd ../../

cd externals
build_path=../../../../build_cpu/build_externals/metis/install ./download_dependencies.sh

echo "========================"
echo "    METIS BUILD DONE    "
echo "========================"

cd ../build_cpu/build_externals/kokkos/build

cmake -DCMAKE_INSTALL_PREFIX=./install \
    -DKokkos_ENABLE_OPENMP=ON \
    ../../../../externals/kokkos
make -j install

echo "========================"
echo "   KOKKOS BUILD DONE    "
echo "========================"

cd ../../kokkos-kernels/build

cmake -DCMAKE_INSTALL_PREFIX=./install -DKokkos_ROOT=../../kokkos/build/install \
    ../../../../externals/kokkos-kernels
make -j install

echo "================================"
echo "   KOKKOS-KERNELS BUILD DONE    "
echo "================================"


cd ../../../

cmake -DGPU_BUILD=0 ..
make -j"${nthreads}"

echo "========================"
echo "   DG-MPI BUILD DONE    "
echo "========================"
