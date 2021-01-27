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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import tensorflow as tf
import numpy as np
from planet import tools
from planet.training import define_summaries
from planet.training import utility
import os


def define_model(data, trainer, config):
    tf.logging.info('Build TensorFlow compute graph.')
    dependencies = []
    cleanups = []
    step = trainer.step
    global_step = trainer.global_step
    phase = trainer.phase

    # Instantiate network blocks.
    cell = config.cell()
    kwargs = dict(create_scope_now_=True)
    encoder = tf.make_template('encoder', config.encoder, **kwargs)
    heads = tools.AttrDict(_unlocked=True)
    dummy_features = cell.features_from_state(cell.zero_state(1, tf.float32))
    for key, head in config.heads.items():
        name = 'head_{}'.format(key)
        kwargs = dict(create_scope_now_=True)
        if key in data:
            kwargs['data_shape'] = data[key].shape[2:].as_list()
        elif key == 'action_target':
            kwargs['data_shape'] = data['action'].shape[2:].as_list()
        heads[key] = tf.make_template(name, head, **kwargs)
        if key == 'image' and config.aug == 'rad' and config.aug_same == False:
            print('image aug feature')
            input_feature = tf.concat([dummy_features, tf.zeros([1, 2], dtype=tf.float32)], -1)
        else:
            input_feature = dummy_features
        heads[key](input_feature)  # Initialize weights.

    # Apply and optimize model.
    embedded = encoder(data)
    with tf.control_dependencies(dependencies):
        embedded = tf.identity(embedded)
    graph = tools.AttrDict(locals())
    prior, posterior = tools.unroll.closed_loop(
        cell, embedded, data['action'], config.debug)
    objectives, cstr_pct = utility.compute_objectives(
        posterior, prior, data, graph, config, trainer)
    summaries, grad_norms = utility.apply_optimizers(
        objectives, trainer, config)

    # Active data collection.
    with tf.variable_scope('collection'):
        with tf.control_dependencies(summaries):  # Make sure to train first.
            for name, params in config.train_collects.items():
                schedule = tools.schedule.binary(
                    step, config.batch_shape[0],
                    params.steps_after, params.steps_every, params.steps_until)
                summary, _ = tf.cond(
                    tf.logical_and(tf.equal(trainer.phase, 'train'), schedule),
                    functools.partial(
                        utility.simulate_episodes, config, params, graph, cleanups,
                        expensive_summaries=False, gif_summary=False, name=name),
                    lambda: (tf.constant(''), tf.constant(0.0)),
                    name='should_collect_' + name)
                summaries.append(summary)

    # Compute summaries.
    graph = tools.AttrDict(locals())
    summary, score = tf.cond(
        trainer.log,
        lambda: define_summaries.define_summaries(graph, config, cleanups),
        lambda: (tf.constant(''), tf.zeros((0,), tf.float32)),
        name='summaries')
    summaries = tf.summary.merge([summaries, summary])
    dependencies.append(utility.print_metrics(
        {ob.name: ob.value for ob in objectives},
        step, config.print_metrics_every, 'objectives'))
    dependencies.append(utility.print_metrics(
        grad_norms, step, config.print_metrics_every, 'grad_norms'))
    with tf.control_dependencies(dependencies):
        score = tf.identity(score)
    return score, summaries, cleanups


def reward_statistics(pred, tar, logdir):
    name = 'contra_step'

    def save_reward(pred, tar):
        path = os.path.join(logdir, name + '.npy')
        est = pred.reshape(-1)
        gd = tar.reshape(-1)
        # est = pred.numpy().reshape(-1)
        # gd = tar.numpy().reshape(-1)
        # est = tf.make_ndarray(pred).reshape(-1)
        # gd = tf.make_ndarray(tar).reshape(-1)
        batch = np.stack((gd, est)).T
        if os.path.exists(path):
            prev = np.load(path)
            new = np.concatenate((prev, batch))
        else:
            new = batch
        print(new.shape)
        if new.shape[0] > 1005:
            exit()
        np.save(os.path.join(logdir, name), new)
        return float(0.0)

    op = tf.py_func(save_reward, inp=[pred, tar], Tout=tf.float64)
    print(op)
    return op


def define_testmodel(data, trainer, config, logdir):
    tf.logging.info('Build TensorFlow compute graph.')
    dependencies = []
    cleanups = []
    step = trainer.step
    global_step = trainer.global_step
    phase = trainer.phase

    # Instantiate network blocks.
    cell = config.cell()
    kwargs = dict(create_scope_now_=True)
    encoder = tf.make_template('encoder', config.encoder, **kwargs)
    heads = tools.AttrDict(_unlocked=True)
    dummy_features = cell.features_from_state(cell.zero_state(1, tf.float32))
    for key, head in config.heads.items():
        name = 'head_{}'.format(key)
        kwargs = dict(create_scope_now_=True)
        if key in data:
            kwargs['data_shape'] = data[key].shape[2:].as_list()
        elif key == 'action_target':
            kwargs['data_shape'] = data['action'].shape[2:].as_list()
        heads[key] = tf.make_template(name, head, **kwargs)
        heads[key](dummy_features)  # Initialize weights.
    print(cell, encoder,)
    # Apply and optimize model.
    embedded = encoder(data)
    with tf.control_dependencies(dependencies):
        embedded = tf.identity(embedded)
    graph = tools.AttrDict(locals())
    prior, posterior = tools.unroll.closed_loop(
        cell, embedded, data['action'], config.debug)

    features = graph.cell.features_from_state(posterior)
    pred = heads['reward'](features)
    # dependencies.append(reward_statistics(pred, data['reward'], logdir))
    summaries = []
    with tf.variable_scope('simulation'):
        sim_returns = []
        for name, params in config.test_collects.items():
            # These are expensive and equivalent for train and test phases, so only
            # do one of them.
            print(name, params)
            sim_summary, score = tf.cond(
                tf.equal(graph.phase, 'test'),
                lambda: utility.simulate_episodes(
                    config, params, graph, cleanups,
                    expensive_summaries=True,
                    gif_summary=True,
                    name=name),
                lambda: ('', 0.0),
                name='should_simulate_' + params.task.name)
    # with tf.variable_scope('collection'):
    #     with tf.control_dependencies(summaries):  # Make sure to train first.
    #         for name, params in config.train_collects.items():
    #             # schedule = tools.schedule.binary(
    #             #     step, config.batch_shape[0],
    #             #     params.steps_after, params.steps_every, params.steps_until)
    #             # summary, _ = tf.cond(
    #             #     tf.logical_and(tf.equal(trainer.phase, 'train'), schedule),
    #             #     functools.partial(
    #             #         utility.simulate_episodes, config, params, graph, cleanups,
    #             #         expensive_summaries=True, gif_summary=False, name=name),
    #             #     lambda: (tf.constant(''), tf.constant(0.0)),
    #             #     name='should_collect_' + name)
    #             summary, score = utility.simulate_episodes(config, params, graph, cleanups,
    #                                                    expensive_summaries=False, gif_summary=False, name=name)
    #             # dependencies.append(summary)

    # print('wuuw', sim_return)
    # objectives = utility.compute_objectives(
    #     posterior, prior, data, graph, config, trainer)
    # summaries, grad_norms = utility.apply_optimizers(
    #     objectives, trainer, config)

    # # Active data collection.
    # with tf.variable_scope('collection'):
    #   with tf.control_dependencies(summaries):  # Make sure to train first.
    #     for name, params in config.train_collects.items():
    #       schedule = tools.schedule.binary(
    #           step, config.batch_shape[0],
    #           params.steps_after, params.steps_every, params.steps_until)
    #       summary, _ = tf.cond(
    #           tf.logical_and(tf.equal(trainer.phase, 'train'), schedule),
    #           functools.partial(
    #               utility.simulate_episodes, config, params, graph, cleanups,
    #               expensive_summaries=True, gif_summary=False, name=name),
    #           lambda: (tf.constant(''), tf.constant(0.0)),
    #           name='should_collect_' + name)
    #       summaries.append(summary)

    # # Compute summaries.
    # score = tf.zeros((0,), tf.float32)
    # summary, score = tf.cond(
    #     trainer.log,
    #     lambda: define_summaries.define_summaries(graph, config, cleanups),
    #     lambda: (tf.constant(''), tf.zeros((0,), tf.float32)),
    #     name='summaries')
    # summaries = tf.summary.merge([summaries, summary])
    # dependencies.append(utility.print_metrics(
    #     {ob.name: ob.value for ob in objectives},
    #     step, config.print_metrics_every, 'objectives'))
    with tf.control_dependencies(dependencies):
        score = tf.identity(score)
    return score, summaries, cleanups
