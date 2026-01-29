#!/bin/bash
# Script per compilare il simulatore Ising

# Carica moduli necessari (adatta al tuo cluster)
# module load mpi
# module load gcc

# Compila con ottimizzazioni
mpicxx -std=c++17 -O3 -fopenmp -DUSE_PHILOX -I./include src/main.cpp -o ising_sim.exe

if [ $? -eq 0 ]; then
    echo "Compilazione completata: ising_sim.exe"
else
    echo "Errore di compilazione!"
    exit 1
fi
