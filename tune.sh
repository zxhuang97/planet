#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --partition=GPU
#SBATCH --exclude=compute-0-[7,9]
#SBATCH --mail-type=END,FAIL             # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=zixuanhu@cs.cmu.edu     # Where to send mail
#SBATCH -o ./log_tune2.txt
#SBATCH -e ./err_tune2.txt
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=3
#SBATCH --time=16-20:00:
#SBATCH --mem=150G
#SBATCH --gres=gpu:3

set -o
set -x
set -u
set -e

source ~/anaconda3/etc/profile.d/conda.sh
conda activate tf1


python3 -m planet.scripts.rayTune --logdir cheetah_run  --params "{tasks: [cheetah_run],r_loss: contra,contra_h: 12, contra_unit: traj, reward_loss_scale: 10.0}" --num_runs 1000 --resume_runs False



