#!/bin/bash
#SBATCH -c 4
#SBATCH --mem 32767
#SBATCH -p gpu-highvram
#SBATCH -G 1
#SBATCH --constraint=l40s
#SBATCH -t 20:00:00
#SBATCH -o logs/grads-%j.out

# Same tickets as job-ticket-sizes-results.sh, plus dense-training gradients:
# first 500 optimizer steps + 500 uniformly sampled from all 46,900 (bf16, ~50 MB/model at h=32)
GRAD_EARLY=500
GRAD_SAMPLE=500

nvidia-smi
module load conda/latest
conda activate torchlth
python /work/pi_jensen_umass_edu/sthiagarajam_umass_edu/lth-reimp/lth-efficiency/lottery-find-tickets.py -e 50 -r 10 -p 0.2 -t 15 -s 32 -d cuda --grad-early $GRAD_EARLY --grad-sample $GRAD_SAMPLE
