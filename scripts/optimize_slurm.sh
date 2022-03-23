#!/bin/bash 
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --tasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00

module load python/3.8.10
module load gcc/8.4.0
module load cuda/10.2
module load cudacore/.11.1.1
module load cudnn/8.2.0
source /home/user_account/pytorch/bin/activate
srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 unzip -qq /home/user_account/projects/def-account/user_account/sbu_dataset.zip -d $SLURM_TMPDIR/sbu_dataset
srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 mkdir -p $SLURM_TMPDIR/sbu_dataset/folds_cache

srun python /home/user_account/scratch/Key-Actor-Detection/optimization/ax_hyperparameter_optimization.py /home/user_account/scratch/Key-Actor-Detection/configs/sbu.yaml "dataset_path=$SLURM_TMPDIR/sbu_dataset" "training.save_dir=null" "caching.folds_cache_path=$SLURM_TMPDIR/sbu_dataset/folds_cache"