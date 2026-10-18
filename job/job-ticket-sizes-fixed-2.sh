#!/bin/bash
#SBATCH -c 4
#SBATCH --mem 32767
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH --constraint=a100|l40s
#SBATCH -t 19:00:00
#SBATCH -o logs/results-%A-%a.out
#SBATCH --array=0-7

P=(16 32 64 128 256 300 512 1024)
PR=(0.0221 0.0876 0.1487 0.2057 0.2589 0.2705 0.3085 0.3548)

nvidia-smi
module load conda/latest
conda activate torchlth
python /work/pi_jensen_umass_edu/sthiagarajam_umass_edu/lth-reimp/lth-efficiency/lottery-find-tickets.py -e 50 -r 10 -p ${PR[$SLURM_ARRAY_TASK_ID]} -t 15 -s ${P[$SLURM_ARRAY_TASK_ID]} -d cuda
