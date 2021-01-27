#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --partition=GPU
#SBATCH --exclude=compute-0-[7,9,11]
#SBATCH --mail-type=END,FAIL             # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=zixuanhu@cs.cmu.edu     # Where to send mail
#SBATCH -o ./llog3.txt
#SBATCH -e ./eerr3.txt
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





#python3 -m planet.scripts.train --logdir log/cartpole_swingup/baseline3 --params '{tasks: [cartpole_swingup]}' --num_runs 5 --resume_runs False


#python3 -m planet.scripts.train --logdir log/finger_spin/baseline3 --params '{tasks: [finger_spin]}' --num_runs 5 --resume_runs False


#python3 -m planet.scripts.train --logdir log/cheetah_run/baseline3 --params '{tasks: [cheetah_run]}' --num_runs 5 --resume_runs False

python3 -m planet.scripts.train --logdir log/cup_catch/baseline3 --params '{tasks: [cup_catch]}' --num_runs 6 --resume_runs False

#python3 -m planet.scripts.train --logdir log/walker_walk/baseline3 --params '{tasks: [walker_walk]}' --num_runs 5 --resume_runs False

#python3 -m planet.scripts.train --logdir log/reacher_easy/baseline3 --params '{tasks: [reacher_easy]}' --num_runs 5 --resume_runs False




