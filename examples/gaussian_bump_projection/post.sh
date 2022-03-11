#!/bin/bash

# Run solver with mpiexec and pass the number of nodes as an argument
#EXEC="mpiexec -n $1 ../../build_gpu/src/post/post"
EXEC="mpiexec -n $1 ../../build_cpu/src/post/post"

FLAGS="$FLAGS" # put flags here

CMD="$EXEC $FLAGS"

echo
echo $CMD
echo
$CMD |& tee log.out
