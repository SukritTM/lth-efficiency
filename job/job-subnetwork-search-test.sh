#!/bin/bash
#SBATCH -c 4
#SBATCH --mem 32767
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH --constraint=l40s
#SBATCH -t 2:00:00
#SBATCH -o logs/results-%j.out
#SBATCH --qos=short

nvidia-smi
module load conda/latest
conda activate torchlth
python /work/pi_jensen_umass_edu/sthiagarajam_umass_edu/lth-reimp/lth-efficiency/lottery-reinit-experiment.py -n 420 -d cuda -i /work/pi_jensen_umass_edu/sthiagarajam_umass_edu/lth-reimp/lth-efficiency/experiment_data/subnetworks-e50-r10-p0.4383-t15-s1024.pkl