#!/bin/bash
#SBATCH -p bme_gpu
#SBATCH -J 0cerwei
#SBATCH -t 120:00:00
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=gpu:NVIDIAA10080GBPCIe:1
#SBATCH -o /home_data/home/jianght2023/projects/cervical-cancer/run_res/50_0gong_yang_weight_res50_job%j.out
#SBATCH -e /home_data/home/jianght2023/projects/cervical-cancer/run_res/50_0gong_yang_weight_res50_job%j.out

echo ${SLURM_JOB_NODELIST}
echo start on $(date)

source /home_data/home/jianght2023/miniconda3/etc/profile.d/conda.sh

cd /home_data/home/jianght2023/projects/cervical-cancer

conda activate pytorch && python train.py --train-csv total_train_yang_yin.csv --valid-csv total_test_yang_yin.csv --positive-csv total_pos_supp_yang.csv  --batch-size 16  --img-batch 50 --epochs 300 --res-savedir /public/home/jianght2023/pths_resnet/50_0gong_lt_res50 --arch resnet50 --res-weights /public/home/jianght2023/pths2/resnet2/resnet.pth --weights /public/home/jianght2023/50_pths_0gong_weight_res50_supyang/model-None-44.pth --where /public/home/jianght2023/50_pths_0gong_weight_res50_supyangyin --lr-head 3e-6 --lr-res 1e-7 --lr 1e-7 --num-classes 2 --loss-fns CELoss  --long-tails True --multi-tasks 1  --alpha 0.2 --tasks label --backbone vit_res  --reduction none --cont --backbone vit_res --logdir weight1_gong_50_weight_yangyin.txt --loss-weights 10

echo end on $(date)