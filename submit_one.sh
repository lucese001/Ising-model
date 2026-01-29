#!/bin/bash
# Script per sottomettere un singolo job
# Uso: ./submit_one.sh <L>
# Esempio: ./submit_one.sh 8

if [ -z "$1" ]; then
    echo "Uso: $0 <L>"
    echo "  L = 8, 16, 32, 64, o 128"
    exit 1
fi

L=$1

# Controlla che L sia valido
if [[ ! "$L" =~ ^(8|16|32|64|128)$ ]]; then
    echo "Errore: L deve essere 8, 16, 32, 64, o 128"
    exit 1
fi

# Controlla che l'eseguibile esista
if [ ! -f "ising_sim.exe" ]; then
    echo "Errore: ising_sim.exe non trovato. Esegui prima ./compile.sh"
    exit 1
fi

# Crea directory necessarie
mkdir -p output logs

echo "Sottomissione job per L=$L..."
qsub jobs/job_L${L}.sh

echo "Job sottomesso! Usa 'qstat' per controllare lo stato."
