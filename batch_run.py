import os

script_name = 'resample'
# run_name = 'baseline5'
# run_name = 'weighted'
run_name = 'rank_margin'
# envs = ['finger_spin']
# envs = ['walker_walk', 'cup_catch']
# envs = ['finger_spin', 'cartpole_swingup', 'reacher_easy', 'cheetah_run']
# envs = ['finger_spin', 'cartpole_swingup', 'reacher_easy', 'cheetah_run', 'walker_walk', 'cup_catch']
envs = ['cartpole_swingup']
num_runs = 5
repeat = 1

# rank_margin
params = "'r_loss: contra, contra_h: 12, contra_unit: rank, epoch: 50, " \
         "reward_loss_scale: 10.0, margin: 1'"


# #aug_base
# params = "'epoch: 50, aug: rad, aug_same: False'"
#

# rank
# params = "'r_loss: contra, contra_h: 12, contra_unit: rank, epoch: 105, " \
#          "reward_loss_scale: 10.0'"

#weighted
# params = "'r_loss: contra, contra_h: 12, contra_unit: weighted, epoch: 105, " \
#          "reward_loss_scale: 10.0, temp: 100.0'"


# # aug10
# params = "'r_loss: contra, contra_h: 12, contra_unit: traj, epoch: 105, " \
#          "reward_loss_scale: 10.0, aug: rad, aug_same: False '"

# baseline
# params = "'epoch: 105'"
# resample
# params = "'r_loss: contra, contra_h: 12, contra_unit: traj, " \
#          "reward_loss_scale: 10.0, divergence_scale: 1.0, num_units: 250'"
# params = "'r_loss: contra, contra_h: 12, contra_unit: traj, " \
#          "reward_loss_scale: 10.0, divergence_scale: 1.0, num_units: 300, iter_ep: 800'"
# params = "'r_loss: contra, contra_h: 12, contra_unit: traj, " \
#          "reward_loss_scale: 3.0, divergence_scale: 1.0'"


# resample + aug
# params = "'r_loss: contra, contra_h: 12, contra_unit: weighted, " \
#          "reward_loss_scale: 10.0, aug: rad, aug_same: False'"
# params = "'r_loss: contra, contra_h: 12, contra_unit: traj, epoch: 105, " \
#          "reward_loss_scale: 10.0, aug: rad, aug_same: False'"
# params = "'r_loss: contra, contra_h: 12, contra_unit: traj, " \
#          "reward_loss_scale: 10.0, aug: drq, aug_same: True, iter_ep: 1000'"



# resample + aug + simclr
# params = "'r_loss: contra, contra_h: 12, contra_unit: simclr, " \
#          "reward_loss_scale: 10.0, aug: rad, aug_same: False,  batch_shape: [64, 50]'"

# more eval
# params = "'r_loss: contra, contra_h: 12, contra_unit: traj, " \
#          "epoch: 20, iter_ep: 5000, test_steps: 500, checkpoint_every: 1000, " \
#          "num_clt_epoch: 50'"


for i in range(repeat):
    for env in envs:
        cmd = 'sbatch {}.sh {} {} {} {}'.format(script_name, run_name,
                                                env, params, num_runs)
        # os.system('cat {} benchmark/{}/{}/params.txt'.fo)
        os.system(cmd)
