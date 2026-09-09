#!/bin/bash
#SBATCH -c 4
#SBATCH --mem 32767
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH --constraint=l40s
#SBATCH -t 1:00:00
#SBATCH -o logs/results-%A-%a.out
#SBATCH --array=0-7

P=(8 16 32 64 128 256 300 512)

nvidia-smi
module load conda/latest
conda activate torchlth
python /work/pi_jensen_umass_edu/sthiagarajam_umass_edu/lth-reimp/lth-efficiency/lottery-reinit-experiment.py -n 5 -d cuda -i /work/pi_jensen_umass_edu/sthiagarajam_umass_edu/lth-reimp/lth-efficiency/experiment_data/subnetworks-e50-r10-p0.2000-t15-s${P[$SLURM_ARRAY_TASK_ID]}.pkl