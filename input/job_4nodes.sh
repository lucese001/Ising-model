#!/bin/bash
#PBS -N ising_4nodes
#PBS -l nodes=4:ppn=24
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -o logs/ising_4nodes.log

cd $PBS_O_WORKDIR

# Configuration: 4 nodes, 8 MPI ranks (2 per node), 12 threads each
export OMP_NUM_THREADS=12
export OMP_PROC_BIND=close
export OMP_PLACES=cores

mpirun -n 8 ./ising_old
