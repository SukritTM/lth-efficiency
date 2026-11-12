#!/bin/bash
#SBATCH -c 4
#SBATCH --mem 32767
#SBATCH -p gpu-highvram
#SBATCH -G 1
#SBATCH --constraint=l40s
#SBATCH -t 4:00:00
#SBATCH -o logs/results-%A-%a.out
#SBATCH --array=1-15

nvidia-smi
module load conda/latest
conda activate torchlth
python /work/pi_jensen_umass_edu/sthiagarajam_umass_edu/lth-reimp/lth-efficiency/lottery-find-tickets-pmnist.py -e 50 -r 10 -p 0.2000 -t 1 -s 32 -b $SLURM_ARRAY_TASK_ID -d cuda
