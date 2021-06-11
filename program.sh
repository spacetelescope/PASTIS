#!/bin/bash
#SBATCH -J full_DM         # run's name
#SBATCH -N 1                   # request 1 node 
#SBATCH -c 16                 # request 8 cpu per task
#SBATCH --mem=16GB             # request 16GB
#SBATCH -t 6:00:00            # request 6 hours walltime
#SBATCH -o Out.txt             # output file name
#SBATCH -e Err.txt             # error file name
#SBATCH --mail-type=BEGIN,END  # send me a mail at beginning and end of the job
#SBATCH --mail-user=david.bourgeois@lam.fr

python /home/dbourgeois/GitHub/PASTIS/pastis/launchers/run_partial_rst.py
