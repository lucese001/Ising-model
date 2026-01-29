#!/bin/bash
# Script per sottomettere tutti i job

# Crea directory necessarie
mkdir -p output logs

# Controlla che l'eseguibile esista
if [ ! -f "ising_sim.exe" ]; then
    echo "Errore: ising_sim.exe non trovato. Esegui prima ./compile.sh"
    exit 1
fi

echo "Sottomissione job per Ising 2D..."

# Sottometti tutti i job
qsub jobs/job_L8.sh
qsub jobs/job_L16.sh
qsub jobs/job_L32.sh
qsub jobs/job_L64.sh
qsub jobs/job_L128.sh

echo ""
echo "Job sottomessi! Usa 'qstat' per controllare lo stato."
echo ""
echo "Output files saranno in:"
echo "  output/meas_L8.txt"
echo "  output/meas_L16.txt"
echo "  output/meas_L32.txt"
echo "  output/meas_L64.txt"
echo "  output/meas_L128.txt"
