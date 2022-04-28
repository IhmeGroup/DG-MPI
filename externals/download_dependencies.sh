#!/bin/bash

set -e

# Download a file and check that the file passes a hash key check
download_file () {
    name=$1
    url=$2
    hash_key=$3
    # Download file
    wget $url --no-check-certificate
    # Check that the hash matches the expected value
    if [ $(sha512sum $name | grep -Eo '^\w+') == $hash_key ]; then
        echo $name passed hash key check
    else
        echo $name failed hash key check!!!
    fi
}

# Number of threads to use for compiling
nthreads=$(getconf _NPROCESSORS_ONLN)

# -- METIS -- #
# Download
name=metis-5.1.0.tar.gz
url=http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis/metis-5.1.0.tar.gz
hash_key=deea47749d13bd06fbeaf98a53c6c0b61603ddc17a43dae81d72c8015576f6495fd83c11b0ef68d024879ed5415c14ebdbd87ce49c181bdac680573bea8bdb25
download_file $name $url $hash_key

# Unzip
mkdir -p metis
tar -xvzf $name -C metis --strip-components 1
rm $name

# Compile
cd metis

make config cc=gcc prefix=$build_path
make install -j"${nthreads}"
cd ..

# -- PHDF5 -- #
echo "========================"
echo "  BUILDING HDF5"
echo "========================"
echo $PWD

wget https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.13/hdf5-1.13.0/src/hdf5-1.13.0.tar.gz
tar xf hdf5-1.13.0.tar.gz
mv hdf5-1.13.0 hdf5-download
cd hdf5-download
CC=`which mpicc` CXX=`which mpic++` ./configure --prefix=${PWD}/install --enable-parallel --enable-cxx=no --enable-hl=yes --enable-hltools=yes --enable-tools=yes --enable-parallel-tools=yes --enable-java=no --enable-tests=no --enable-shared=yes --enable-static=yes
make -j && make -j install
cd ..
