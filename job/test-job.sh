#!/bin/bash
#SBATCH -c 4
#SBATCH --mem 16384
#SBATCH -p cpu
#SBATCH -c 4
#SBATCH -t 3:30:00
#SBATCH -o logs/test-%j.out

nvidia-smi
module load conda/latest
conda activate torchlth
python /work/pi_jensen_umass_edu/sthiagarajam_umass_edu/lth-reimp/lth-efficiency/test-experiment.py -e 2 -s 32 -d cpu