#!/bin/bash
#SBATCH --partition=vgpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=gan
##SBATCH --ntasks=1
#SBATCH --time=3:00:00 # 3 hours
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sampollard888@gmail.com
