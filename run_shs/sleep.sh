#!/bin/bash
#SBATCH -p bme_quick
#SBATCH -J distri
#SBATCH -t 3:00:00
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=gpu:NVIDIAA10080GBPCIe:2
#SBATCH -o /home_data/home/jianght2023/projects/cervical-cancer/run_shs/50_moe_job%j.out
#SBATCH -e /home_data/home/jianght2023/projects/cervical-cancer/run_shs/50_moe_job%j.out

echo ${SLURM_JOB_NODELIST}
echo start on $(date)

source /home_data/home/jianght2023/miniconda3/etc/profile.d/conda.sh

cd /home_data/home/jianght2023/projects/cervical-cancer/run_shs

conda activate pytorch && python sleep.py
echo end on $(date)