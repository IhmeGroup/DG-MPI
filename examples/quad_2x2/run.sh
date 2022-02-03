#!/bin/bash

# Run solver with mpiexec and pass the number of nodes as an argument
EXEC="../../externals/mpich/install/bin/mpiexec -n $1 ../../build/src/exec/main"

FLAGS="$FLAGS" # put flags here

CMD="$EXEC $FLAGS"

echo
echo $CMD
echo
$CMD |& tee log.out
