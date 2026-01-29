#!/bin/bash
#PBS -N ising_L128
#PBS -l nodes=1:ppn=12
#PBS -l walltime=168:00:00
#PBS -o logs/ising_L128.out
#PBS -e logs/ising_L128.err

cd $PBS_O_WORKDIR

# Crea directory output e logs se non esistono
mkdir -p output logs

# Copia parametri specifici per L=128
cp input/params_L128.txt input/dimensioni.txt

# Esegui simulazione (1 MPI rank, 12 threads)
export OMP_NUM_THREADS=12
mpirun -np 1 ./ising_sim.exe

# Salva output con nome specifico
mv output/meas.txt output/meas_L128.txt

echo "Simulazione L=128 completata"
