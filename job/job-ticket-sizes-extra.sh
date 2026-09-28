#!/bin/bash
#SBATCH -c 4
#SBATCH --mem 32767
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH --constraint=l40s
#SBATCH -t 20:00:00
#SBATCH -o logs/results-%A-%a.out


nvidia-smi
module load conda/latest
conda activate torchlth
python /work/pi_jensen_umass_edu/sthiagarajam_umass_edu/lth-reimp/lth-efficiency/lottery-find-tickets.py -e 50 -r 10 -p 0.1487 -t 15 -s 16} -d cuda