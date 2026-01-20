#!/bin/bash
#PBS -N ising_2nodes
#PBS -l nodes=2:ppn=24
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -o logs/ising_2nodes.log

cd $PBS_O_WORKDIR

# Configuration: 2 nodes, 4 MPI ranks (2 per node), 12 threads each
export OMP_NUM_THREADS=12
export OMP_PROC_BIND=close
export OMP_PLACES=cores

mpirun -n 4 ./ising_old
