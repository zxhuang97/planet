#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --partition=GPU
#SBATCH --exclude=compute-0-[7,9,11]
#SBATCH --mail-type=END,FAIL             # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=zixuanhu@cs.cmu.edu     # Where to send mail
#SBATCH -o ./log.txt
#SBATCH -e ./err.txt
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --time=10-00:00:
#SBATCH --mem=90G
#SBATCH --gres=gpu:1
set -x
set -u
set -e

source ~/anaconda3/etc/profile.d/conda.sh
conda activate tf1

NAME=$1
ENV=$2
PARAM=$3
RUNS=$4

python3 -m planet.scripts.train --logdir log/$ENV/$NAME --params "{tasks: [${ENV}], ${PARAM}}" --num_runs $RUNS --resume_runs False


#python3 -m planet.scripts.train --logdir log/cup_catch/aug  --params "{tasks: [cup_catch],aug: True,  r_loss: contra,contra_h: 12, contra_unit: traj, reward_loss_scale: 10.0}" --num_runs 5 --resume_runs False
