#!/bin/bash 
#SBATCH --nodes=2
#SBATCH --gres=gpu:2
#SBATCH --tasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00

module load python/3.8.10
module load gcc/8.4.0
module load cuda/10.2
module load cudacore/.11.1.1
module load cudnn/8.2.0
source /home/rohitram/pytorch/bin/activate

srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 unzip -qq /home/rohitram/projects/def-mudl/rohitram/sbu_dataset.zip -d $SLURM_TMPDIR/sbu_dataset

srun python /home/rohitram/scratch/Key-Actor-Detection/run.py /home/rohitram/scratch/Key-Actor-Detection/configs/sbu.yaml "dataset_path=$SLURM_TMPDIR/sbu_dataset" "training.save_dir=$SLURM_TMPDIR/mlruns" 

cp -r $SLURM_TMPDIR/mlruns /home/rohitram/scratch




