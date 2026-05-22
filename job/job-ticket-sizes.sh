#!/bin/bash
#SBATCH -c 4
#SBATCH --mem 32767
#SBATCH -p gpu
#SBATCH --constraint=a100-80g
#SBATCH -t 12:00:00
#SBATCH -o logs/results-%j.out
#SBATCH --array=0-7

P=(8 16 32 64 128 256 512 1024)

nvidia-smi
module load conda/latest
conda activate torchlth
python /work/pi_jensen_umass_edu/sthiagarajam_umass_edu/lth-reimp/lth-efficiency/lottery-find-tickets.py -e 50 -r 10 -p 0.2 -t 15 -s ${P[$SLURM_ARRAY_TASK_ID]} -d cpu