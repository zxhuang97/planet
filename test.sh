#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --partition=GPU
#SBATCH --exclude=compute-0-[7,9,11,13]
#SBATCH --mail-type=END,FAIL             # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=zixuanhu@cs.cmu.edu     # Where to send mail
#SBATCH -o ./llog13.txt
#SBATCH -e ./eerr13.txt
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=2
#SBATCH --time=03-00:00:
#SBATCH --mem=90G
#SBATCH --gres=gpu:1

set -x
set -u
set -e

source ~/anaconda3/etc/profile.d/conda.sh
conda activate tf1
#python3 -m planet.scripts.train --logdir log --params '{tasks: [cheetah_run]}' --num_runs 11 --resume_runs False

#python3 -m planet.scripts.train --logdir log/margin_contra --params "{tasks: [cheetah_run],r_loss: contra,reward_loss_scale: 10.0}" --num_runs 2 --resume_runs False

#python3 -m planet.scripts.train --logdir log/hard_negative --params "{tasks: [cheetah_run],r_loss: contra,reward_loss_scale: 10.0,loader: hard,train_steps: 0}" --num_runs 1 --resume_runs True


python3 -m planet.scripts.test --logdir log/hard_negative --params "{tasks: [cheetah_run],r_loss: contra,reward_loss_scale: 10.0,loader: hard,train_steps: 0,planner: cem_eval}" --num_runs 1 --resume_runs True

