#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --partition=GPU
#SBATCH --exclude=compute-0-[7,9]
#SBATCH --mail-type=END,FAIL             # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=zixuanhu@cs.cmu.edu     # Where to send mail
#SBATCH -o ./llog2.txt
#SBATCH -e ./eerr2.txt
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --time=05-00:00:
#SBATCH --mem=90G
#SBATCH --gres=gpu:1

set -x
set -u
set -e

source ~/anaconda3/etc/profile.d/conda.sh
conda activate tf1
#python3 -m planet.scripts.train --logdir log --params '{tasks: [cheetah_run]}' --num_runs 5 --resume_runs False

#python3 -m planet.scripts.train --logdir log/margin_contra --params "{tasks: [cheetah_run],r_loss: contra,reward_loss_scale: 10.0}" --num_runs 2 --resume_runs False

#python3 -m planet.scripts.train --logdir log/reg --params "{tasks: [cheetah_run],r_loss: l2,reward_loss_scale: 1.0}" --num_runs 5 --resume_runs False
#python3 -m planet.scripts.train --logdir log/cup_catch/hard_negative --params "{tasks: [cup_catch],r_loss: contra,reward_loss_scale: 10.0,loader: hard}" --num_runs 3 --resume_runs True

python3 -m planet.scripts.train --logdir log/test  --params "{tasks: [reacher_easy],r_loss: contra,contra_h: 12, contra_unit: traj, reward_loss_scale: 10.0}" --num_runs 500 --resume_runs False


#python3 -m planet.scripts.train --logdir log/reacher_easy/aug  --params "{tasks: [reacher_easy],aug: True,  r_loss: contra,contra_h: 12, contra_unit: traj, reward_loss_scale: 10.0}" --num_runs 3 --resume_runs False

#python3 -m planet.scripts.train --logdir log/hard_resample  --params "{tasks: [cheetah_run],r_loss: contra,contra_h: 12, contra_unit: traj, reward_loss_scale: 1.0, resample: 10, hr: 0.1}" --num_runs 3 --resume_runs False
