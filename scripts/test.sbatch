#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --mem=20GB
#SBATCH --job-name=hpc4473C7
#SBATCH --output=b433_init_normal.out

module purge
module load anaconda3/2020.07
eval "$(conda shell.bash hook)"
conda activate harienv

python ../src/main.py -e 200 -pf b433_init_normal -ini normal
