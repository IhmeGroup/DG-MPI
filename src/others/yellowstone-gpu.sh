#!/bin/bash
# Yellowstone specific build instructions (GPU version) - updated as of 2/14/22

# This file should be called from the 'DG-MPI/src/others' folder.

# NOTE: Make sure you follow directions to install cmake version 3.22 on the
# github page (see externals/README.md) and then run this command as follows:

# cmake=<PATH_TO_CMAKE>/cmake-3.22.2/bin/cmake ./yellowstone-gpu.sh

# Load necessary modules
module purge
module load cuda
module load gnu8/8.3.0
#module load hdf5/1.10.5
module load mpich/3.3.1

if [[ -z "${cmake}" ]]; then
    export cmake=`which cmake`
fi

set -e

# move to the home dir
cd ../../

git submodule update --init

mkdir -p build_gpu/build_externals
cd build_gpu/build_externals
mkdir metis
mkdir -p kokkos/build/install
mkdir -p kokkos-kernels/build/install
cd ../../

cd externals
build_path=../../../../build_gpu/build_externals/metis/install ./download_dependencies.sh

echo "========================"
echo "    METIS BUILD DONE    "
echo "========================"

cd ../build_gpu/build_externals/kokkos/build

$cmake  -DCMAKE_INSTALL_PREFIX=./install -DKokkos_ENABLE_OPENMP=ON \
    -DKokkos_ENABLE_CUDA=ON -DKokkos_ENABLE_CUDA_LAMBDA=ON \
    -DKokkos_CXX_STANDARD=17 \
    -DCMAKE_CXX_COMPILER=${PWD}/../../../../externals/kokkos/bin/nvcc_wrapper \
    ../../../../externals/kokkos
make -j install

echo "========================"
echo "   KOKKOS BUILD DONE    "
echo "========================"

cd ../../kokkos-kernels/build

$cmake -DCMAKE_INSTALL_PREFIX=./install \
    -DKokkos_ROOT=../../kokkos/build/install \
    -DKokkos_CXX_STANDARD=17 \
    -DCMAKE_CXX_COMPILER=${PWD}/../../kokkos/build/install/bin/nvcc_wrapper \
    -DKokkosKernels_REQUIRE_DEVICES=CUDA \
    ../../../../externals/kokkos-kernels
make -j install

echo "================================"
echo "   KOKKOS-KERNELS BUILD DONE    "
echo "================================"


cd ../../../
$cmake -DGPU_BUILD=1 \
    -DCMAKE_CXX_COMPILER=${PWD}/build_externals/kokkos/build/install/bin/nvcc_wrapper ..
make -j"${nthreads}"

echo "========================"
echo "   DG-MPI BUILD DONE    "
echo "========================"
