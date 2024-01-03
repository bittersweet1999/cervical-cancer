#!/bin/bash
#SBATCH -p bme_gpu
#SBATCH -J 0cerwei
#SBATCH -t 120:00:00
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=gpu:NVIDIAA10080GBPCIe:1
#SBATCH -o /home_data/home/jianght2023/projects/cervical-cancer/run_res/50_moe_job%j.out
#SBATCH -e /home_data/home/jianght2023/projects/cervical-cancer/run_res/50_moe_job%j.out

echo ${SLURM_JOB_NODELIST}
echo start on $(date)

source /home_data/home/jianght2023/miniconda3/etc/profile.d/conda.sh

cd /home_data/home/jianght2023/projects/cervical-cancer

conda activate pytorch && python train2.py --train-csv select_train.csv --valid-csv total_test2.csv --positive-csv total_pos_supp_yang.csv  --batch-size 8  --img-batch 50 --epochs 300 --res-savedir /public/home/jianght2023/pths_resnet/50_0gong_lt_res50 --arch resnet50 --res-weights /public/home/jianght2023/pths2/resnet2/resnet.pth --where /public/home/jianght2023/50_pths_moe2 --lr-head 3e-4 --lr-res 1e-4 --lr 1e-4 --num-classes 2,3,4,2,2,2 --loss-fns CELoss  --long-tails False --multi-tasks 6  --alpha 0.2 --tasks label --backbone vit_moe  --reduction none --cont --backbone vit_moe --logdir weight1_ttt.txt --loss-weights 10
echo end on $(date)