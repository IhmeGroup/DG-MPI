#!/bin/bash

#SBATCH --job-name="vortexp2"
#SBATCH --output="log_%j.out"
#SBATCH --nodes=1
#SBATCH --time=10:00:00
#SBATCH --partition=compute

export OMP_NUM_THREADS=32
EXEC="../../../../build_cpu/src/exec/main"

NRANK=1

# General flags
FLAGS="$FLAGS --kokkos-num-devices=$NRANK"

CMD="mpirun -n $NRANK" 

#CMD="nsys profile -o nsys --trace cuda,mpi --force-overwrite true mpirun -np $NRANK"
CMD="$CMD $EXEC $FLAGS"

echo
echo $CMD
echo
$CMD
