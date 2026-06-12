#!/bin/bash
#SBATCH -c 4
#SBATCH --mem 32767
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH --constraint=l40s
#SBATCH -t 15:00:00
#SBATCH -o logs/results-%A-%a.out
#SBATCH --array=0-7

P=(8 16 32 64 128 256 512 1024)

nvidia-smi
module load conda/latest
conda activate torchlth
python /work/pi_jensen_umass_edu/sthiagarajam_umass_edu/lth-reimp/lth-efficiency/lottery-find-tickets-pmnist.py -e 50 -r 10 -p 0.073 -t 15 -s ${P[$SLURM_ARRAY_TASK_ID]} -d cuda