#!/bin/bash

# point this variable to the solver binary on your system
EXEC="../../build/src/exec/solver"

NCPU=0 # number of nodes

FLAGS="$FLAGS" # put flags here

CMD="$EXEC $FLAGS"

echo
echo $CMD
echo
$CMD |& tee log.out
