#!/bin/bash
#SBATCH -N 1
#SBATCH --exclusive 
#SBATCH --partition=GPUS
#SBATCH -o salida/output.txt
#SBATCH -e salida/errors.txt
./heat $1 $2
