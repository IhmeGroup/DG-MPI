#!/bin/bash

# If testing with MPI, use 2 ranks
if [ $1 == mpi ]; then
    prefix=""
    n=2
else
    prefix="-"
    n=1
fi

# List of tests that need MPI, with colons between
mpi_tests="Mesh*:"

# Run tests
mpiexec -n ${n} $(dirname ${BASH_SOURCE[0]})/../../build/test/test_gtest_all \
    --gtest_filter=${prefix}${mpi_tests}
