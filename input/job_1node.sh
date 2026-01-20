#!/bin/bash
#PBS -N ising_1node
#PBS -l nodes=1:ppn=24
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -o logs/ising_1node.log

cd $PBS_O_WORKDIR

# Configuration: 1 node, 2 MPI ranks, 12 threads each (24 total cores)
export OMP_NUM_THREADS=12
export OMP_PROC_BIND=close
export OMP_PLACES=cores

mpirun -n 2 ./ising_old
