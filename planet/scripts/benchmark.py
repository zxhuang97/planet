# Copyright 2019 The PlaNet Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Test a Deep Planning Network agent.

Full testing run:
python3 -m planet.scripts.benchmark --logdir benchmark --params "{train_steps: 0, max_steps: 1e7, train_action_noise: 0.0}" \
--num_runs 5 --resume_runs True

For debugging:

python3 -m planet.scripts.train \
    --logdir /path/to/logdir \
    --resume_runs False \
    --num_runs 1000 \
    --config debug \
    --params '{tasks: [cheetah_run]}'
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import argparse
import functools
import os
import sys
import dowel
from dowel import logger, tabular
#
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

# Need offline backend to render summaries from within tf.py_func.
import matplotlib

matplotlib.use('Agg')

import ruamel.yaml as yaml
import tensorflow as tf

from planet import tools
from planet import training
from planet.scripts import configs

import os


def process(logdir, args):
    with args.params.unlocked:
        args.params.logdir = logdir
    config = tools.AttrDict()
    with config.unlocked:
        config = getattr(configs, args.config)(config, args.params)
    tf.reset_default_graph()
    dataset = tools.numpy_episodes.numpy_episodes(
        config.train_dir, config.test_dir, config.batch_shape,
        reader=config.data_reader,
        loader=config.data_loader,
        num_chunks=config.num_chunks,
        preprocess_fn=config.preprocess_fn,
        aug_fn=config.aug_fn)
    for score in training.utility.test(
            training.define_testmodel, dataset, logdir, config):
        yield score


def clean(logdir, methods):
    import glob
    for m in methods:
        path = os.path.join(logdir, m)
        runs = glob.glob(path + '/*/DONE')


def check_finish(base_dir, stages, methods, envs, num):
    finish = True
    print('check if all runs are finished')

    for pref, chkpt in stages.items():
        for method in methods:
            for env in envs:
                for r in range(num):
                    p = os.path.join(base_dir, env, method, '00{}'.format(r + 1), '{}.meta'.format(chkpt))
                    if not os.path.exists(p):
                        print(p)
                        finish = False
    return finish


def main(args):
    import dowel
    from dowel import logger, tabular
    training.utility.set_up_logging()
    stages = {'500k': 'model.ckpt-2502500'}
    # stages = {'1000k': 'model.ckpt-5005000'}
    num_traj = 10
    # stages = {'100k': 'model.ckpt-500500', '500k': 'model.ckpt-2502500'}
    # stages = {'1M': 'model.ckpt-5005000'}
    # stages = {'final': 'model.ckpt-2652650'}
    # stages = {'100k': 'model.ckpt-500500', '500k': 'model.ckpt-2502500', '1M': 'model.ckpt-5005000'}
    # stages = {'100k': 'model.ckpt-600500', '500k': 'model.ckpt-3002500',
    #           'final':'model.ckpt-3182650'}
    # methods = ['weighted_100']
    # methods = ['aug7']
    methods = ['baseline3']
    base_dir = 'benchmark'
    envs = ['cup_catch']
    # envs = ['walker_walk', 'cup_catch']
    # envs = ['finger_spin', 'cartpole_swingup','cheetah_run', 'cup_catch']
    # envs = ['finger_spin', 'cartpole_swingup', 'reacher_easy', 'cheetah_run']
    # envs = ['cartpole_swingup', 'cheetah_run', 'walker_walk', 'cup_catch']
    # envs = ['finger_spin', 'cartpole_swingup', 'reacher_easy', 'cheetah_run', 'walker_walk', 'cup_catch']
    if not check_finish(base_dir, stages, methods, envs, args.num_runs):
        exit()

    for pref, chkpt in stages.items():
        print(pref, 'begin')
        logger.add_output(dowel.StdOutput())
        logger.add_output(dowel.CsvOutput('benchmark_{}.csv'.format(pref)))
        for env in envs:
            tabular.record('Env', env)
            for method in methods:
                means, stds, all_scores = [], [], []
                with args.params.unlocked:
                    args.params.chkpt = chkpt
                    args.params.tasks = [env]
                    args.params.train_action_noise = 0.0
                    # args.params.test_steps = 100000
                    args.params.planner_horizon = 12
                    args.params.eval_ratio = 1/num_traj
                    # args.params.num_units = 200
                    # args.params.r_loss = 'contra'
                    # args.params.aug = 'rad'
                    # args.params.planner = 'cem_eval'
                    # args.params.planner = 'cem'
                    args.params.planner = 'dual1'
                    # args.params.planner = 'sim'

                experiment = training.Experiment(
                    os.path.join(base_dir, env, method),
                    process_fn=functools.partial(process, args=args),
                    num_runs=args.num_runs,
                    ping_every=args.ping_every,
                    resume_runs=args.resume_runs,
                    planner=args.params.planner,
                    task_str=env
                )
                for i,run in enumerate(experiment):
                    scores = []
                    for i, unused_score in enumerate(run):
                        print('unused', unused_score)
                        scores.append(unused_score)
                        if i == num_traj-1:
                            break
                    means.append(np.mean(scores))
                    stds.append(np.std(scores))
                    all_scores.append(scores)
                    print(means)
                    # if args.params.planner != 'cem':
                    #     exit()
                    if args.params.planner == 'cem_eval':
                        np.save(os.path.join(args.logdir, env, method,
                                             '00{}/scores_{}_cem.npy'.format(i, pref)), np.array(all_scores))
                mean, std = np.mean(means), np.std(means)
                print('{}    {}+/-{}'.format(method, int(mean), int(std)))
                if mean > 0:
                    tabular.record(method, '{}+/-{}'.format(int(mean), int(std)))
                if args.params.planner == 'cem_eval':
                    np.save(os.path.join(args.logdir, env, method, 'scores_{}.npy'.format(pref)), np.array(all_scores))
            logger.log(tabular)
            logger.dump_all()
        logger.remove_all()


if __name__ == '__main__':
    boolean = lambda x: bool(['False', 'True'].index(x))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--logdir', required=True)
    parser.add_argument(
        '--num_runs', type=int, default=1)
    parser.add_argument(
        '--config', default='default',
        help='Select a configuration function from scripts/configs.py.')
    parser.add_argument(
        '--params', default='{}',
        help='YAML formatted dictionary to be used by the config.')
    parser.add_argument(
        '--ping_every', type=int, default=0,
        help='Used to prevent conflicts between multiple workers; 0 to disable.')
    parser.add_argument(
        '--resume_runs', type=boolean, default=True,
        help='Whether to resume unfinished runs in the log directory.')
    args_, remaining = parser.parse_known_args()
    args_.params = tools.AttrDict(yaml.safe_load(args_.params.replace('#', ',')))
    args_.logdir = args_.logdir and os.path.expanduser(args_.logdir)
    remaining.insert(0, sys.argv[0])
    tf.app.run(lambda _: main(args_), remaining)
