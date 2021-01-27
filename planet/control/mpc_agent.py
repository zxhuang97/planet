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

from tensorflow_probability import distributions as tfd
import tensorflow as tf

from planet.tools import nested
from six.moves import cPickle as pickle
import os
import numpy as np


class MPCAgent(object):

    def __init__(self, batch_env, step, is_training, should_log, config):
        self._batch_env = batch_env
        self._step = step  # Trainer step, not environment step.
        self._is_training = is_training
        self._should_log = should_log
        self._config = config
        self._cell = config.cell
        self._length = 0
        state = self._cell.zero_state(len(batch_env), tf.float32)
        var_like = lambda x: tf.get_local_variable(
            x.name.split(':')[0].replace('/', '_') + '_var',
            shape=x.shape,
            initializer=lambda *_, **__: tf.zeros_like(x), use_resource=True)
        self._state = nested.map(var_like, state)
        self._prev_action = tf.get_local_variable(
            'prev_action_var', shape=self._batch_env.action.shape,
            initializer=lambda *_, **__: tf.zeros_like(self._batch_env.action),
            use_resource=True)

    def begin_episode(self, agent_indices):
        a = tf.print('reset everything')

        self._length = 0
        state = nested.map(
            lambda tensor: tf.gather(tensor, agent_indices),
            self._state)
        reset_state = nested.map(
            lambda var, val: tf.scatter_update(var, agent_indices, 0 * val),
            self._state, state, flatten=True)
        reset_prev_action = self._prev_action.assign(
            tf.zeros_like(self._prev_action))
        with tf.control_dependencies(reset_state + (reset_prev_action, a)):
            return tf.constant('')

    def perform(self, agent_indices, observ, env_state=None):
        self._length = self._length + 1
        if self._config.aug_fn is not None:
            print('augmented agent')
            observ = self._config.aug_fn({'image': observ}, phase='plan')['image']
        observ = self._config.preprocess_fn(observ)

        embedded = self._config.encoder({'image': observ[:, None]})[:, 0]
        state = nested.map(
            lambda tensor: tf.gather(tensor, agent_indices),
            self._state)

        prev_action = self._prev_action + 0
        with tf.control_dependencies([prev_action]):
            use_obs = tf.ones(tf.shape(agent_indices), tf.bool)[:, None]
            _, state = self._cell((embedded, prev_action, use_obs), state)
        #
        # a = tf.print(env_state)
        # with tf.control_dependencies([a]):
        #     prev_action = self._prev_action + 0

        action = self._config.planner(
            self._cell, self._config.objective, state,
            embedded.shape[1:].as_list(),
            prev_action.shape[1:].as_list(), env_state=env_state)
        # (1, 12, 2)
        action = action[:, 0]
        if self._config.exploration:
            scale = self._config.exploration.scale
            if self._config.exploration.schedule:
                scale *= self._config.exploration.schedule(self._step)
            action = tfd.Normal(action, scale).sample()

        action = tf.clip_by_value(action, -1, 1)
        # a = tf.print('action ', tf.reduce_max(action), tf.reduce_min(action))
        # with tf.control_dependencies([a]):
        remember_action = self._prev_action.assign(action)
        remember_state = nested.map(
            lambda var, val: tf.scatter_update(var, agent_indices, val),
            self._state, state, flatten=True)
        with tf.control_dependencies(remember_state + (remember_action,)):
            return tf.identity(action), tf.constant('')

    def experience(self, agent_indices, *experience):
        return tf.constant('')

    def end_episode(self, agent_indices):
        return tf.constant('')


import uuid
import datetime
import io


class rolloutSaver(object):
    def __init__(self, outdir):
        self._outdir = outdir and os.path.expanduser(outdir)
        self._episode = []
        self._length = 0

    def reset(self):
        print('the saver is reset')
        self._episode = []
        self._length = 0
        return True

    def _get_filename(self):
        timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
        identifier = str(uuid.uuid4()).replace('-', '')
        filename = '{}-{}.npz'.format(timestamp, identifier)
        filename = os.path.join(self._outdir, filename)
        return filename

    def _get_episode(self):
        episode = {k: [t[k] for t in self._episode] for k in self._episode[0]}
        episode = {k: np.array(v) for k, v in episode.items()}
        for key, sequence in episode.items():
            if sequence.dtype == 'object':
                message = "Sequence '{}' is not numeric:\n{}"
                raise RuntimeError(message.format(key, sequence))
        return episode

    def _process_observ(self, observ):
        if not isinstance(observ, dict):
            observ = {'observ': observ}
        return observ

    def _write(self, episode, filename):
        if not tf.gfile.Exists(self._outdir):
            tf.gfile.MakeDirs(self._outdir)
        with io.BytesIO() as file_:
            np.savez_compressed(file_, **episode)
            file_.seek(0)
            with tf.gfile.Open(filename, 'w') as ff:
                ff.write(file_.read())
        folder = os.path.basename(self._outdir)
        name = os.path.splitext(os.path.basename(filename))[0]
        print('Recorded rollout {} to {}.'.format(name, folder))

    # def _process_step(self, action, observ, reward, done, info):
    def _process_step(self, action, done, observ, collect, all_action, all_reward):
        transition = self._process_observ(observ).copy()
        transition['action'] = action
        transition['all_action'] = all_action
        transition['collect'] = collect
        transition['all_reward'] = all_reward
        # if np.sum(all_reward)==0 and np.sum(collect)!=0:
        #     print('?????')
        #     exit()
        self._length += 1
        print('process step  {}'.format(self._length))

        self._episode.append(transition)
        if done:
            print('rollout finished, length {}'.format(self._length))
            episode = self._get_episode()
            if self._outdir:
                filename = self._get_filename()
                self._write(episode, filename)
        return True


class dualMPCAgent1(object):

    def __init__(self, batch_env, step, is_training, should_log, config):
        self._batch_env = batch_env
        self._step = step  # Trainer step, not environment step.
        self._is_training = is_training
        self._should_log = should_log
        self._config = config
        self._cell = config.cell
        self._length = 0
        self.num_episodes = 0
        self.saver = rolloutSaver(os.path.join(config.logdir, 'rollout'))

        state = self._cell.zero_state(len(batch_env), tf.float32)
        var_like = lambda x: tf.get_local_variable(
            x.name.split(':')[0].replace('/', '_') + '_var',
            shape=x.shape,
            initializer=lambda *_, **__: tf.zeros_like(x), use_resource=True)
        self._state = nested.map(var_like, state)
        self._prev_action = tf.get_local_variable(
            'prev_action_var', shape=self._batch_env.action.shape,
            initializer=lambda *_, **__: tf.zeros_like(self._batch_env.action),
            use_resource=True)

    def begin_episode(self, agent_indices):
        self._length = 0
        a = tf.print('reset everything')
        r = tf.py_func(self.saver.reset, inp=[], Tout=tf.bool)
        state = nested.map(
            lambda tensor: tf.gather(tensor, agent_indices),
            self._state)
        reset_state = nested.map(
            lambda var, val: tf.scatter_update(var, agent_indices, 0 * val),
            self._state, state, flatten=True)
        reset_prev_action = self._prev_action.assign(
            tf.zeros_like(self._prev_action))
        with tf.control_dependencies(reset_state + (reset_prev_action, r, a)):
            return tf.constant('')


    def perform(self, agent_indices, ori_observ, env_state=None):
        observ = ori_observ + 0
        self._length = self._length + 1
        if self._config.aug_fn is not None:
            print('augmented agent')
            observ = self._config.aug_fn({'image': observ}, phase='plan')['image']

        observ = self._config.preprocess_fn(observ)

        embedded = self._config.encoder({'image': observ[:, None]})[:, 0]
        state = nested.map(
            lambda tensor: tf.gather(tensor, agent_indices),
            self._state)

        prev_action = self._prev_action + 0
        with tf.control_dependencies([prev_action]):
            use_obs = tf.ones(tf.shape(agent_indices), tf.bool)[:, None]
            _, state = self._cell((embedded, prev_action, use_obs), state)

        action, collect, all_action, all_reward = self._config.planner(
            self._cell, self._config.objective, state,
            embedded.shape[1:].as_list(),
            prev_action.shape[1:].as_list(), env_state=env_state)
        action = action[:, 0]
        if self._config.exploration:
            scale = self._config.exploration.scale
            if self._config.exploration.schedule:
                scale *= self._config.exploration.schedule(self._step)
            action = tfd.Normal(action, scale).sample()

        action = tf.clip_by_value(action, -1, 1)
        # a = tf.print('action ', tf.reduce_max(action), tf.reduce_min(action))
        # with tf.control_dependencies([a]):
        remember_action = self._prev_action.assign(action)
        remember_state = nested.map(
            lambda var, val: tf.scatter_update(var, agent_indices, val),
            self._state, state, flatten=True)
        with tf.control_dependencies(remember_state + (remember_action,)):
            return tf.identity(action), (collect, all_action, all_reward)

    def experience(self, agent_indices, action, done, observ, step_summary):
        a = tf.print('experience')
        with tf.control_dependencies([a]):
            collect, all_action, all_reward = step_summary
        r = tf.py_func(self.saver._process_step, inp=[action, done, observ, collect, all_action, all_reward],
                       Tout=tf.bool)
        with tf.control_dependencies([r]):
            return r

    def end_episode(self, agent_indices):
        return tf.constant('')


from planet.tools import numpy_episodes


class fileEnv(object):
    def __init__(self, directory):
        self.directory = directory
        self.all_epi = numpy_episodes.reload_loader(numpy_episodes.episode_reader, self.directory)
        self.ind_epi = 0

        print('file env created')

    def step(self):
        for epi in self.all_epi:
            self.ind_step = 0
            print(epi['all_reward'])
            for obs, action, all_action, collect, all_reward in zip(epi['observ'], epi['action'],
                                                                    epi['all_action'], epi['collect'],
                                                                    epi['all_reward']):
                yield obs, action, all_action, collect, all_reward


def gener(directory):
    all_epi = numpy_episodes.reload_loader(numpy_episodes.episode_reader, directory)
    ind_epi = 0
    for epi in all_epi:
        ind_stp = 0
        for obs, action, all_action, collect, all_reward in zip(epi['observ'], epi['action'],
                                                                epi['all_action'], epi['collect'],
                                                                epi['all_reward']):
            ind_stp+=1
            print('s', ind_epi, ind_stp)
            yield {'observ': obs, 'action': action, 'all_action': all_action,
                   'collect': collect, 'all_reward':all_reward}
        ind_epi+=1


class dualMPCAgent2(object):

    def __init__(self, batch_env, step, is_training, should_log, config):
        print('mpc agent dual2')
        self._batch_env = batch_env
        self._step = step  # Trainer step, not environment step.
        self._is_training = is_training
        self._should_log = should_log
        self._config = config
        self._cell = config.cell
        self._length = 0
        self.logdir = config.logdir
        self.rival_dir = os.path.join('benchmark', config.rival, 'rollout')
        state = self._cell.zero_state(len(batch_env), tf.float32)
        var_like = lambda x: tf.get_local_variable(
            x.name.split(':')[0].replace('/', '_') + '_var',
            shape=x.shape,
            initializer=lambda *_, **__: tf.zeros_like(x), use_resource=True)
        self._state = nested.map(var_like, state)
        self._prev_action = tf.get_local_variable(
            'prev_action_var', shape=self._batch_env.action.shape,
            initializer=lambda *_, **__: tf.zeros_like(self._batch_env.action),
            use_resource=True)
        import functools
        dtypes, shapes = numpy_episodes._read_spec2(numpy_episodes.episode_reader, self.rival_dir)
        rival = tf.data.Dataset.from_generator(
            functools.partial(gener, self.rival_dir),
            dtypes, shapes)

        def wh(sq):
            print('sq', sq['observ'])
            return sq['observ'], sq['action'], sq['all_action'], sq['collect'], sq['all_reward']

        rival = rival.map(wh)
        rival = rival.batch(1)
        self.rival = rival.make_one_shot_iterator()
        # for obs, action, all_action, collect, all_reward in gener(self.rival_dir):
        #     continue

    def begin_episode(self, agent_indices):
        a = tf.print('reset everything')

        self._length = 0
        state = nested.map(
            lambda tensor: tf.gather(tensor, agent_indices),
            self._state)
        reset_state = nested.map(
            lambda var, val: tf.scatter_update(var, agent_indices, 0 * val),
            self._state, state, flatten=True)
        reset_prev_action = self._prev_action.assign(
            tf.zeros_like(self._prev_action))
        with tf.control_dependencies(reset_state + (reset_prev_action, a)):
            return tf.constant('')

    def gd_save(self, batch):
        name = 'onrival'
        path = os.path.join(self.logdir, name + '.npy')
        def eval(batch):
            #10, 1, 1000, 3
            batch = np.squeeze(batch)
            batch = np.expand_dims(batch, 0)
            if os.path.exists(path):
                prev = np.load(path)
                new = np.concatenate((prev, batch), 0)
            else:
                new = batch
            print(new.shape)
            np.save(os.path.join(self.logdir, name), new)
            return True
        op = tf.py_func(eval, inp=[batch], Tout=tf.bool)
        return op

    def perform(self, agent_indices, observ, env_state=None):
        observ, action, all_action, collect, all_reward = self.rival.get_next()
        print('wtf', observ, action,  all_action, collect, all_reward)
        observ, all_action, all_reward = tf.squeeze(observ, 0), tf.squeeze(all_action, 0), tf.squeeze(all_reward, 0)
        self._length = self._length + 1
        if self._config.aug_fn is not None:
            print('augmented agent')
            observ = self._config.aug_fn({'image': observ}, phase='plan')['image']
        observ = self._config.preprocess_fn(observ)

        embedded = self._config.encoder({'image': observ[:, None]})[:, 0]
        state = nested.map(
            lambda tensor: tf.gather(tensor, agent_indices),
            self._state)

        prev_action = self._prev_action + 0
        with tf.control_dependencies([prev_action]):
            use_obs = tf.ones(tf.shape(agent_indices), tf.bool)[:, None]
            _, state = self._cell((embedded, prev_action, use_obs), state)

        triple = self._config.planner(
            self._cell, self._config.objective, state, all_action, all_reward,
            embedded.shape[1:].as_list(),
            prev_action.shape[1:].as_list(), env_state=env_state)

        save = tf.cond(collect[0][0],
                lambda: self.gd_save(triple),
                lambda: tf.no_op())

        with tf.control_dependencies([save]):
            action = action[0]

        # a = tf.print('action ', tf.reduce_max(action), tf.reduce_min(action))
        # with tf.control_dependencies([a]):
        remember_action = self._prev_action.assign(action)
        remember_state = nested.map(
            lambda var, val: tf.scatter_update(var, agent_indices, val),
            self._state, state, flatten=True)
        with tf.control_dependencies(remember_state + (remember_action,)):
            return tf.identity(action), tf.constant('')

    def experience(self, agent_indices, *experience):
        return tf.constant('')

    def end_episode(self, agent_indices):
        return tf.constant('')
