#!/bin/bash
#PBS -N ising_L8
#PBS -l nodes=1:ppn=2
#PBS -l walltime=02:00:00
#PBS -o logs/ising_L8.out
#PBS -e logs/ising_L8.err

cd $PBS_O_WORKDIR

# Setup OpenMPI
unset LD_LIBRARY_PATH
export MPI_ROOT=/storage/local/exp_soft/local_sl7/mpi/openmpi-4.0.5
export PATH=$MPI_ROOT/bin:$PATH
export LD_LIBRARY_PATH=$MPI_ROOT/lib

# Crea directory output e logs se non esistono
mkdir -p output logs

# Copia parametri specifici per L=8
cp input/params_L8.txt input/dimensioni.txt

# Esegui simulazione (1 MPI rank, 2 threads)
export OMP_NUM_THREADS=2
mpirun -np 1 ./ising_sim.exe

# Salva output con nome specifico
mv output/meas.txt output/meas_L8.txt

echo "Simulazione L=8 completata"
