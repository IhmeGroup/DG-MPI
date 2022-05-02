#!/bin/bash

#SBATCH --job-name="vortex"
#SBATCH --output="log_%j.out"
#SBATCH --nodes=1
#SBATCH --time=10:00:00
#SBATCH -p gpu-k40                # gpu partition
##SBATCH --partition=compute

export OMP_NUM_THREADS=32
module load cuda
EXEC="../../../../build_gpu/src/exec/main"

NRANK=1
CMD="mpiexec -n $NRANK"
#CMD="mpiexec -n $NRANK nvprof"
CMD="$CMD $EXEC $FLAGS"

echo
echo $CMD
echo
$CMD
