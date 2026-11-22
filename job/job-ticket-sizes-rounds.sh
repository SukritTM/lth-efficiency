#!/bin/bash
#SBATCH -c 4
#SBATCH --mem 32767
#SBATCH -p gpu-highvram
#SBATCH -G 1
#SBATCH --constraint=l40s
#SBATCH -t 8:00:00
#SBATCH -o logs/results-%A-%a.out
#SBATCH --array=0-134

P=(8 16 32 64 128 256 300 512 1024)
R=(9 15 22 28 35 42 43 48 55)
NUM_SEEDS=15

W=$((SLURM_ARRAY_TASK_ID / NUM_SEEDS))
B=$((SLURM_ARRAY_TASK_ID % NUM_SEEDS + 1))

nvidia-smi
module load conda/latest
conda activate torchlth
python /work/pi_jensen_umass_edu/sthiagarajam_umass_edu/lth-reimp/lth-efficiency/lottery-find-tickets.py -e 50 -r ${R[$W]} -p 0.1000 -t 1 -s ${P[$W]} -b $B -d cuda
