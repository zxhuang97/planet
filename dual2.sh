#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --partition=GPU
#SBATCH --exclude=compute-0-[9]
#SBATCH --mail-type=END,FAIL             # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=zixuanhu@cs.cmu.edu     # Where to send mail
#SBATCH -o ./llog4.txt
#SBATCH -e ./eerr4.txt
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --time=03-00:00:
#SBATCH --mem=60G
#SBATCH --gres=gpu:1

set -x
set -u
set -e

source ~/anaconda3/etc/profile.d/conda.sh
conda activate tf1


#python3 -m planet.scripts.benchmark --logdir benchmark --params "{train_steps: 0, max_steps: 1e7, train_action_noise: 0.0, planner: dual1}" --num_runs 5 --resume_runs True

python3 -m planet.scripts.dual2 --logdir benchmark --params "{train_steps: 0, max_steps: 1e7, train_action_noise: 0.0, planner: dual2}" --num_runs 1 --resume_runs True

#python3 -m planet.scripts.benchmark --logdir benchmark --params "{train_steps: 0, max_steps: 1e7, train_action_noise: 0.0}" --num_runs 5 --resume_runs True
