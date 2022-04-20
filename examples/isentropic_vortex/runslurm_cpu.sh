#!/bin/bash

#SBATCH --job-name="vortexp2"
#SBATCH --output="log_%j.out"
#SBATCH --nodes=1
#SBATCH --time=10:00:00
#SBATCH --partition=compute

EXEC="/home/ihme/bornhoft/utilities/dg_dev/dg-mpi/build_cpu/src/exec/main"

NRANK=1
NCPU=0
NUTIL=1
NOCPU=1
NOTHR=32
CSIZE=2000
RSIZE=0


# General flags
FLAGS="$FLAGS -logfile logs/log_%.log"
#FLAGS="$FLAGS -hdf5:forcerw" # needed for HDF5 outputs

# Memory flags
#FLAGS="$FLAGS -ll:csize $CSIZE" # system memory
#FLAGS="$FLAGS -ll:rsize $RSIZE" # registered memory (not used at the moment)

# CPU flags
#FLAGS="$FLAGS -ll:cpu $NCPU"
#FLAGS="$FLAGS -ll:util $NUTIL"

# OpenMP flags
#FLAGS="$FLAGS -ll:onuma 0"
# FLAGS="$FLAGS -ll:ht_sharing 0"
#FLAGS="$FLAGS -ll:ocpu $NOCPU"
#FLAGS="$FLAGS -ll:othr $NOTHR"

# Additional flags
# FLAGS="$FLAGS -ll:show_rsrv" # to show core reservation map
# FLAGS="$FLAGS -lg:prof $NRANK -lg:prof_logfile prof_%.gz" # to profile using legion_prof

CMD="mpirun -n $NRANK"
CMD="$CMD $EXEC $FLAGS"

echo
echo $CMD
echo
$CMD
