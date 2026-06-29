#!/bin/bash
#SBATCH -c 4
#SBATCH --mem 32767
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH --constraint=l40s
#SBATCH -t 10:00:00
#SBATCH -o logs/results-%A-%a.out
#SBATCH --array=0-13

P=(5 6 7 8 9 11 12 13 14 15 16 17 18 19)

nvidia-smi
module load conda/latest
conda activate torchlth
python /work/pi_jensen_umass_edu/sthiagarajam_umass_edu/lth-reimp/lth-efficiency/lottery-find-tickets-streams.py -e 50 -r 10 -p 0.2000 -t 15 -s ${P[$SLURM_ARRAY_TASK_ID]} -d cuda