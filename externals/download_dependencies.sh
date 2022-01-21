#!/bin/bash

# Download a file and check that the file passes a hash key check
download_file () {
    name=$1
    url=$2
    hash_key=$3
    # Download file
    wget $url
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
mkdir metis
tar -xvzf $name -C metis --strip-components 1
rm $name

# Compile
cd metis
make config cc=gcc prefix=../install
make install -j"${nthreads}"
