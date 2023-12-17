#!/bin/bash
#SBATCH -p bme_gpu
#SBATCH -J 1cerwei
#SBATCH -t 120:00:00
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=gpu:NVIDIAA10080GBPCIe:1
#SBATCH -o /home_data/home/jianght2023/projects/cervical-cancer/run_res/50_1gong_yang_weight_res34_job%j.out
#SBATCH -e /home_data/home/jianght2023/projects/cervical-cancer/run_res/50_1gong_yang_weight_res34_job%j.out

echo ${SLURM_JOB_NODELIST}
echo start on $(date)

source /home_data/home/jianght2023/miniconda3/etc/profile.d/conda.sh

cd /home_data/home/jianght2023/projects/cervical-cancer

conda activate pytorch && python train.py --train-csv total_train_yang_yin.csv --valid-csv total_test_yang_yin.csv --positive-csv total_pos_supp_yang.csv   --batch-size 16  --img-batch 50  --epochs 300 --res-savedir /public/home/jianght2023/pths_resnet/50_1gong2 --arch resnet34 --res-weights /public/home/jianght2023/pths2/resnet2/resnet.pth  --where /public/home/jianght2023/50_pths_1gong_weight_yangyin --lr-head 3e-14,1e-5,3e-18,5e-13,5e-14 --lr-res 1e-6 --lr 1e-7 --num-classes 2,2,2,4,2 --loss-fns CELoss,CELoss,CELoss,CELoss,CELoss  --weights /home_data/home/jianght2023/projects/model_cell_gong_manyi_level_fungus.pth  --long-tails True,False,True,True,False --multi-tasks 5  --alpha 0.0 --tasks fungus,label,fungus,fungus,fungus --backbone vit_res  --reduction none --cont  --backbone vit_res --logdir gong_weigh2.txt  --loss-weights 10

echo end on $(date)