from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import functools
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

# Need offline backend to render summaries from within tf.py_func.
import matplotlib

matplotlib.use('Agg')

import ruamel.yaml as yaml
import tensorflow as tf
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.suggest.bohb import TuneBOHB
from ray.tune.suggest.bayesopt import BayesOptSearch
from planet import tools
from planet import training
from planet.scripts import configs


# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

def process(logdir, args):
    with args.params.unlocked:
        args.params.logdir = logdir
    config = tools.AttrDict()
    with config.unlocked:
        config = getattr(configs, args.config)(config, args.params)
    training.utility.collect_initial_episodes(config)
    tf.reset_default_graph()
    dataset = tools.numpy_episodes.numpy_episodes(
        config.train_dir, config.test_dir, config.batch_shape,
        reader=config.data_reader,
        loader=config.data_loader,
        num_chunks=config.num_chunks,
        preprocess_fn=config.preprocess_fn,
        aug_fn=config.aug_fn)
    for score in training.utility.train(
            training.define_model, dataset, logdir, config):
        yield score


def main(args):
    def trainable(config):
        print('begin a trial')
        args.params = tools.AttrDict(yaml.safe_load(args.params.replace('#', ',')))
        args.logdir = args.logdir and os.path.expanduser(args.logdir)
        print('debug ', config["divergence_scale"], config["reward_loss_scale"])
        with args.params.unlocked:
            args.params.divergence_scale = config["divergence_scale"]
            args.params.reward_loss_scale = config["reward_loss_scale"]
            # args.params.main_learning_rate = config["main_learning_rate"]
            args.params.test_steps = 50
            # args.params.num_units = config['num_units']
            args.params.test_traj = 5
        training.utility.set_up_logging()
        experiment = training.Experiment(
            args.logdir,
            process_fn=functools.partial(process, args=args),
            num_runs=args.num_runs,
            ping_every=args.ping_every,
            resume_runs=args.resume_runs)
        for run in experiment:
            for test_score in run:
                if test_score > 1.0:
                    tune.report(mean_score=test_score)
            break

    import ConfigSpace as CS
    import ConfigSpace.hyperparameters as CSH
    # search = {
    #     "divergence_scale": tune.quniform(1, 30, 1),
    #     "reward_loss_scale": tune.quniform(1, 50, 1),
    # }
    search = {
        "divergence_scale": tune.grid_search([0.1, 1, 2, 3, 5, 10]),
        "reward_loss_scale": tune.grid_search([1, 2, 5, 10, 20]),
    }
    config_space = CS.ConfigurationSpace(seed=1234)
    config_space.add_hyperparameter(
        CSH.UniformIntegerHyperparameter(name="divergence_scale", lower=1, upper=30))
    config_space.add_hyperparameter(
        CSH.UniformIntegerHyperparameter(name="reward_loss_scale", lower=1, upper=50))
    # config_space.add_hyperparameter(
    #     CSH.UniformFloatHyperparameter("main_learning_rate", lower=0.0001, upper=0.05, log=True))
    config_space.add_hyperparameter(
        CSH.UniformIntegerHyperparameter("main_learning_rate", lower=1, upper=500, log=True))
    config_space.add_hyperparameter(
        CSH.UniformIntegerHyperparameter(name="num_units", lower=150, upper=400, q=50))
    bayesopt = BayesOptSearch(metric="mean_loss", mode="min")
    bohb_hyperband = HyperBandForBOHB(metric="mean_score", mode="max", time_attr="training_iteration", max_t=30,
                                      reduction_factor=3)
    bohb_search = TuneBOHB(space=config_space, max_concurrent=1, metric="mean_score", mode="max")
    bayesopt = BayesOptSearch(max_concurrent=3, metric="mean_score", mode="max")
    asha = ASHAScheduler(metric="mean_score", mode="max", grace_period=6, reduction_factor=3)
    analysis = tune.run(
        trainable,
        config=search,
        num_samples=3,
        scheduler=asha,
        resources_per_trial={"cpu": 16, "gpu": 1},
        stop={"training_iteration": 13},
        # search_alg=bayesopt,
        log_to_file=True
    )
    df = analysis.results_df
    print("Best config: ", analysis.get_best_config(
        metric="mean_score", mode="min"))
    print(df)


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
    remaining.insert(0, sys.argv[0])
    main(args_)
