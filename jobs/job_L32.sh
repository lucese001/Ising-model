#!/bin/bash
#PBS -N ising_L32
#PBS -l nodes=1:ppn=6
#PBS -l walltime=12:00:00
#PBS -o logs/ising_L32.out
#PBS -e logs/ising_L32.err

cd $PBS_O_WORKDIR

# Setup OpenMPI
unset LD_LIBRARY_PATH
export MPI_ROOT=/storage/local/exp_soft/local_sl7/mpi/openmpi-4.0.5
export PATH=$MPI_ROOT/bin:$PATH
export LD_LIBRARY_PATH=$MPI_ROOT/lib

# Crea directory output e logs se non esistono
mkdir -p output logs

# Copia parametri specifici per L=32
cp input/params_L32.txt input/dimensioni.txt

# Esegui simulazione (1 MPI rank, 6 threads)
export OMP_NUM_THREADS=6
mpirun -np 1 ./ising_sim.exe

# Salva output con nome specifico
mv output/meas.txt output/meas_L32.txt

echo "Simulazione L=32 completata"
