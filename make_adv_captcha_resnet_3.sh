#!/bin/sh

#SBATCH -J make_adv_captcha_resnet_3
#SBATCH -o make_adv_captcha_resnet_3.%j.out
#SBATCH -p gpu-2080ti-8
#SBATCH -t 48:00:00

#SBATCH --gres=gpu:4
#SBATCH --nodelist=n17
#SBATCH --ntasks=4
#SBATCH --tasks-per-node=4
#SBATCH --cpus-per-task=1

cd  $SLURM_SUBMIT_DIR

echo "SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR"
echo "CUDA_HOME=$CUDA_HOME"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "CUDA_VERSION=$CUDA_VERSION"

srun -l /bin/hostname
srun -l /bin/pwd
srun -l /bin/date

echo "[ Python Program Start ]"

python3 make_adv_captcha_resnet_3.py

date

squeue  --job  $SLURM_JOBID

echo  "##### END #####"
