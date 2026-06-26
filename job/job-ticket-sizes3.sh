#!/bin/bash
#SBATCH -c 4
#SBATCH --mem 32767
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH --constraint=l40s
#SBATCH -t 10:00:00
#SBATCH -o logs/results-%A-%a.out
#SBATCH --array=8-15

P=(10 20 40 60 80 100 120 140 160 180 200 220 240 260 280 300)

nvidia-smi
module load conda/latest
conda activate torchlth
python /work/pi_jensen_umass_edu/sthiagarajam_umass_edu/lth-reimp/lth-efficiency/lottery-find-tickets-streams.py -e 50 -r 10 -p 0.2000 -t 15 -s ${P[$SLURM_ARRAY_TASK_ID]} -d cuda