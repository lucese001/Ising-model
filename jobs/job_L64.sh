#!/bin/bash
#PBS -N ising_L64
#PBS -l nodes=1:ppn=8
#PBS -l walltime=72:00:00
#PBS -o logs/ising_L64.out
#PBS -e logs/ising_L64.err

cd $PBS_O_WORKDIR

# Crea directory output e logs se non esistono
mkdir -p output logs

# Copia parametri specifici per L=64
cp input/params_L64.txt input/dimensioni.txt

# Esegui simulazione (1 MPI rank, 8 threads)
export OMP_NUM_THREADS=8
mpirun -np 1 ./ising_sim.exe

# Salva output con nome specifico
mv output/meas.txt output/meas_L64.txt

echo "Simulazione L=64 completata"
