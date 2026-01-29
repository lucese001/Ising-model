#!/bin/bash
#PBS -N ising_L16
#PBS -l nodes=1:ppn=4
#PBS -l walltime=04:00:00
#PBS -o logs/ising_L16.out
#PBS -e logs/ising_L16.err

cd $PBS_O_WORKDIR

# Carica moduli necessari
module load mpi/openmpi-x86_64

# Crea directory output e logs se non esistono
mkdir -p output logs

# Copia parametri specifici per L=16
cp input/params_L16.txt input/dimensioni.txt

# Esegui simulazione (1 MPI rank, 4 threads)
export OMP_NUM_THREADS=4
mpirun -np 1 ./ising_sim.exe

# Salva output con nome specifico
mv output/meas.txt output/meas_L16.txt

echo "Simulazione L=16 completata"
