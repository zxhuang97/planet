#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --partition=GPU
#SBATCH --exclude=compute-0-[7,9,11]
#SBATCH --mail-type=END,FAIL             # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=zixuanhu@cs.cmu.edu     # Where to send mail
#SBATCH -o ./llog2.txt
#SBATCH -e ./eerr2.txt
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --time=08-00:00:
#SBATCH --mem=90G
#SBATCH --gres=gpu:1

set -x
set -u
set -e

source ~/anaconda3/etc/profile.d/conda.sh
conda activate tf1

python3 -m planet.scripts.train --logdir log/finger_spin/aug2  --params "{tasks: [finger_spin],  r_loss: contra,contra_h: 12, contra_unit: traj, reward_loss_scale: 10.0, aug: rad, aug_same: True}" --num_runs 5 --resume_runs False


#python3 -m planet.scripts.train --logdir log/cartpole_swingup/aug2  --params "{tasks: [cartpole_swingup],  r_loss: contra,contra_h: 12, contra_unit: traj, reward_loss_scale: 10.0, aug: rad, aug_same: True}" --num_runs 5 --resume_runs False

#python3 -m planet.scripts.train --logdir log/reacher_easy/aug2  --params "{tasks: [reacher_easy],  r_loss: contra,contra_h: 12, contra_unit: traj, reward_loss_scale: 10.0, aug: rad, aug_same: True}" --num_runs 5 --resume_runs False

#python3 -m planet.scripts.train --logdir log/cheetah_run/aug  --params "{tasks: [cheetah_run],aug: True,  r_loss: contra,contra_h: 12, contra_unit: traj, reward_loss_scale: 10.0}" --num_runs 3 --resume_runs False

#python3 -m planet.scripts.train --logdir log/walker_walk/aug  --params "{tasks: [walker_walk],aug: True,  r_loss: contra,contra_h: 12, contra_unit: traj, reward_loss_scale: 10.0}" --num_runs 5 --resume_runs False

#python3 -m planet.scripts.train --logdir log/cup_catch/aug  --params "{tasks: [cup_catch],aug: True,  r_loss: contra,contra_h: 12, contra_unit: traj, reward_loss_scale: 10.0}" --num_runs 5 --resume_runs False
