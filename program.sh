#!/bin/bash
#SBATCH -J DM_24x24         # run's name
#SBATCH -N 1                   # request 1 node 
#SBATCH -c 24                 # request 24 cpu per task
#SBATCH --mem=32GB             # request 16GB
#SBATCH -t 12:00:00            # request 4 hours walltime
#SBATCH -o Out.txt             # output file name
#SBATCH -e Err.txt             # error file name
#SBATCH --mail-type=BEGIN,END  # send me a mail at beginning and end of the job
#SBATCH --mail-user=david.bourgeois@lam.fr

python /home/dbourgeois/GitHub/PASTIS/pastis/launchers/run_partial_rst.py
