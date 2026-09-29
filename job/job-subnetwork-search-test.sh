#!/bin/bash
#SBATCH -c 4
#SBATCH --mem 32767
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH --constraint=l40s
#SBATCH -t 2:00:00
#SBATCH -o logs/results-%A-%a.out
#SBATCH --array=0-7
#SBATCH --qos=short

P=(8 16 32 64 128 256 300 512 1024)
PR=(0.0876 0.1487 0.2057 0.2589 0.3085 0.3548 0.3649 0.3979 0.4383)

nvidia-smi
module load conda/latest
conda activate torchlth
python /work/pi_jensen_umass_edu/sthiagarajam_umass_edu/lth-reimp/lth-efficiency/lottery-reinit-experiment.py -n 10 -d cuda -i /work/pi_jensen_umass_edu/sthiagarajam_umass_edu/lth-reimp/lth-efficiency/experiment_data/subnetworks-e50-r10-p0.4383-t15-s1024.pkl