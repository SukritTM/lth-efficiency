#!/bin/bash
#SBATCH -c 4
#SBATCH --mem 32767
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH --constraint=l40s
#SBATCH -t 10:00:00
#SBATCH -o logs/results-%j.out

nvidia-smi
module load conda/latest
conda activate torchlth
python /work/pi_jensen_umass_edu/sthiagarajam_umass_edu/lth-reimp/lth-efficiency/l1-tickets.py -r 0.9000 -f /work/pi_jensen_umass_edu/sthiagarajam_umass_edu/lth-reimp/lth-efficiency/experiment_data/l1-models-e50-t15-s30.pkl -d cuda