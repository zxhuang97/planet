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

import tensorflow as tf

import functools
from planet import tools
import numpy as np
import dm_env
from dm_control import suite
import multiprocessing
import uuid
import os


def cross_entropy_method(
        cell, objective_fn, state, obs_shape, action_shape, horizon, graph,
        amount=1000, topk=100, iterations=10, min_action=-1, max_action=1, env_state=None):
    obs_shape, action_shape = tuple(obs_shape), tuple(action_shape)
    original_batch = tools.shape(tools.nested.flatten(state)[0])[0]
    initial_state = tools.nested.map(lambda tensor: tf.tile(
        tensor, [amount] + [1] * (tensor.shape.ndims - 1)), state)
    extended_batch = tools.shape(tools.nested.flatten(initial_state)[0])[0]
    use_obs = tf.zeros([extended_batch, horizon, 1], tf.bool)
    obs = tf.zeros((extended_batch, horizon) + obs_shape)

    def iteration(mean_and_stddev, _):
        mean, stddev = mean_and_stddev
        # Sample action proposals from belief.
        normal = tf.random_normal((original_batch, amount, horizon) + action_shape)
        action = normal * stddev[:, None] + mean[:, None]
        action = tf.clip_by_value(action, min_action, max_action)
        # Evaluate proposal actions.
        action = tf.reshape(
            action, (extended_batch, horizon) + action_shape)
        (_, state), _ = tf.nn.dynamic_rnn(
            cell, (0 * obs, action, use_obs), initial_state=initial_state)
        return_ = objective_fn(state)
        return_ = tf.reshape(return_, (original_batch, amount))
        # Re-fit belief to the best ones.
        _, indices = tf.nn.top_k(return_, topk, sorted=False)
        indices += tf.range(original_batch)[:, None] * amount
        best_actions = tf.gather(action, indices)
        mean, variance = tf.nn.moments(best_actions, 1)
        stddev = tf.sqrt(variance + 1e-6)
        return mean, stddev

    mean = tf.zeros((original_batch, horizon) + action_shape)
    stddev = tf.ones((original_batch, horizon) + action_shape)
    if iterations < 1:
        return mean
    mean, stddev = tf.scan(
        iteration, tf.range(iterations), (mean, stddev), back_prop=False)
    mean, stddev = mean[-1], stddev[-1]  # Select belief at last iterations.
    return mean


class ActionRepeat(object):
    def __init__(self, env, amount):
        self._env = env
        self._amount = amount
        self._last_action = None

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        reward = 0
        discount = 1
        for _ in range(self._amount):
            time_step = self._env.step(action)
            reward += time_step.reward
            discount *= time_step.discount
            if time_step.last():
                break
        time_step = dm_env._environment.TimeStep(
            step_type=time_step.step_type,
            reward=reward,
            discount=discount,
            observation=time_step.observation)
        return time_step

    def reset(self, *args, **kwargs):
        self._last_action = None
        return self._env.reset(*args, **kwargs)


class AsyncEvaluator:

    def __init__(self, env_ctor, num_workers):
        self._closed = False
        self._results = {}
        self._requests = multiprocessing.Queue()
        self._responses = multiprocessing.Queue()
        self._workers = [
            multiprocessing.Process(
                target=self._worker,
                args=(env_ctor, self._requests, self._responses))
            for _ in range(num_workers)]
        for worker in self._workers:
            worker.start()
        print('Create the evaluator')

    def __call__(self, state, actions):
        assert not self._closed
        id_ = str(uuid.uuid4())
        self._results[id_] = None
        self._requests.put((id_, state, actions))

        def promise(id_):
            assert id_ in self._results
            while self._results[id_] is None:
                try:
                    other, score = self._responses.get(0.01)
                except multiprocessing.Queue.Empty:
                    continue
                self._results[other] = score
            return self._results.pop(id_)

        return functools.partial(promise, id_)

    def close(self):
        self._closed = True
        for _ in self._workers:
            self._requests.put(None)
        for worker in self._workers:
            worker.join()

    def _worker(self, env_ctor, jobs, results):
        try:
            env = env_ctor()
            while True:
                try:
                    job = jobs.get(0.01)
                    if job is None:
                        break
                    id_, state, actions = job
                except multiprocessing.Queue.Empty:
                    continue
                score = self._evaluate(env, state, actions)
                results.put((id_, score))
        except Exception:
            pass

    def _evaluate(self, env, state, actions):
        env.reset()
        with env.physics.reset_context():
            env.physics.data.qpos[:] = state[0]
            env.physics.data.qvel[:] = state[1]
        score = 0
        scores = []
        for action in actions:
            time_step = env.step(action)
            scores.append(time_step.reward)
        return scores

def create_env(domain, task, repeat):
    env = suite.load(domain, task)
    env = ActionRepeat(env, repeat)
    return env


def gd_eval(state, actions, preds, logdir):

    env_ctor = functools.partial(create_env, domain='cheetah', task='run', repeat=4)
    evaluator = AsyncEvaluator(env_ctor, 40)
    name = 'cem_traj'
    path = os.path.join(logdir, name + '.npy')
    def eval(state, actions, preds):
        promises = []
        for act in actions:
            promise = evaluator(state, act)
            promises.append(promise)
        gds = [promise() for promise in promises]
        gds = np.stack(gds, 0)
        batch = np.stack((gds, preds), 2)
        batch = np.expand_dims(batch, 0)
        if os.path.exists(path):
            prev = np.load(path)
            new = np.concatenate((prev, batch),0)
        else:
            new = batch
        print(new.shape)
        np.save(os.path.join(logdir, name), new)
        return float(0.0)
    op = tf.py_func(eval, inp=[state, actions, preds], Tout=tf.float64)
    return op


def cross_entropy_method_eval(
        cell, objective_fn, state, obs_shape, action_shape, horizon, graph, logdir,
        amount=1000, topk=100, iterations=10, min_action=-1, max_action=1, eval_ratio=0.1, env_state=None):
    obs_shape, action_shape = tuple(obs_shape), tuple(action_shape)
    original_batch = tools.shape(tools.nested.flatten(state)[0])[0]
    initial_state = tools.nested.map(lambda tensor: tf.tile(
        tensor, [amount] + [1] * (tensor.shape.ndims - 1)), state)
    extended_batch = tools.shape(tools.nested.flatten(initial_state)[0])[0]
    use_obs = tf.zeros([extended_batch, horizon, 1], tf.bool)
    obs = tf.zeros((extended_batch, horizon) + obs_shape)

    def iteration(mean_and_stddev, _):
        mean, stddev = mean_and_stddev
        # Sample action proposals from belief.
        normal = tf.random_normal((original_batch, amount, horizon) + action_shape)
        action = normal * stddev[:, None] + mean[:, None]
        action = tf.clip_by_value(action, min_action, max_action)
        # Evaluate proposal actions by latent imagination
        action = tf.reshape(
            action, (extended_batch, horizon) + action_shape)
        (_, state), _ = tf.nn.dynamic_rnn(
            cell, (0 * obs, action, use_obs), initial_state=initial_state)
        reward = objective_fn(state)

        # evaluate candidate trajectories by true simulator
        simu_eval = gd_eval(env_state, action, reward,logdir)
        with tf.control_dependencies([simu_eval]):
            return_ = tf.reduce_sum(reward, 1)

        return_ = tf.reshape(return_, (original_batch, amount))
        # Re-fit belief to the best ones.
        _, indices = tf.nn.top_k(return_, topk, sorted=False)
        indices += tf.range(original_batch)[:, None] * amount
        best_actions = tf.gather(action, indices)
        mean, variance = tf.nn.moments(best_actions, 1)
        stddev = tf.sqrt(variance + 1e-6)
        return mean, stddev

    mean = tf.zeros((original_batch, horizon) + action_shape)
    stddev = tf.ones((original_batch, horizon) + action_shape)

    if iterations < 1:
        return mean
    mean, stddev = tf.scan(
        iteration, tf.range(iterations), (mean, stddev), back_prop=False)
    mean, stddev = mean[-1], stddev[-1]  # Select belief at last iterations.
    return mean
