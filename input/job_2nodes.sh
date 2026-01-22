#!/bin/bash
#PBS -N ising_2nodes
#PBS -l nodes=2:ppn=24
#PBS -l walltime=02:00:00
#PBS -j oe

cd $PBS_O_WORKDIR

# Create logs directory if it doesn't exist
mkdir -p logs

# MPI setup - clear old paths first to avoid conflicts
export MPI_ROOT=/storage/local/exp_soft/local_sl7/mpi/openmpi-4.0.5
export PATH=$MPI_ROOT/bin:$PATH
export LD_LIBRARY_PATH=$MPI_ROOT/lib
unset OPAL_PREFIX

# OpenMP configuration: 2 nodes, 4 MPI ranks, 12 threads each
export OMP_NUM_THREADS=12
export OMP_PROC_BIND=close
export OMP_PLACES=cores

echo "Starting job on $(hostname) at $(date)"
echo "Working directory: $(pwd)"
echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"

mpirun -n 4 ./ising_old 2>&1 | tee logs/ising_2nodes.log

echo "Job finished at $(date)"
