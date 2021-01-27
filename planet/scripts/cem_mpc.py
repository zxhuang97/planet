"""
CEM planning on dm_control tasks using the ground truth dynamics.

Example usage:

  DISABLE_MUJOCO_RENDERING=1 python dm_control_mpc.py cheetah run -r 4 -l 12
"""

import argparse
import collections
import functools
import itertools
import multiprocessing
import uuid
import time
from dm_control import rl
from dm_control import suite
import dm_env
import numpy as np


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
        # time_step = rl.environment.TimeStep(
        #     step_type=time_step.step_type,
        #     reward=reward,
        #     discount=discount,
        #     observation=time_step.observation)
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
        old_state = np.array([env.physics.data.qpos, env.physics.data.qvel])
        # score = 0
        #
        # for action in actions:
        #     time_step = env.step(action)
        #     score += time_step.reward
        # return score
        # scores = []
        # for i, action in enumerate(actions):
        #     time_step = env.step(action)
        #     scores.append(time_step.reward)
        #     if i == 0:
        #         new_state = np.array([env.physics.data.qpos, env.physics.data.qvel])
        # return scores, (old_state, new_state)


def cem_planner(
        evaluator, random, action_spec, state, horizon, proposals, topk, iterations, env_ctor):
    mean = np.zeros((horizon,) + action_spec.shape)
    std = np.ones((horizon,) + action_spec.shape)

    action = random.normal(mean, std)
    for i in range(10):
        test_env = env_ctor()
        test_env.reset()
        with test_env.physics.reset_context():
            test_env.physics.data.qpos[:] = state[0]
            test_env.physics.data.qvel[:] = state[1]
        old_state = np.array([test_env.physics.data.qpos, test_env.physics.data.qvel])
        ts = test_env.step(action[0])
        new_state = np.array([test_env.physics.data.qpos, test_env.physics.data.qvel])
        # print(ts.reward)
        # print(old_state)
        # print(new_state)
    # exit()
    print('1')
    for i in range(iterations):
        actions = []
        promises = []
        for _ in range(proposals):
            action = random.normal(mean, std)
            action = np.clip(action, -1, 1)
            print('initial action', action)
            promise = evaluator(state, action)
            promises.append(promise)
            actions.append(action)
        # scores = [promise() for promise in promises]
        results = [promise() for promise in promises]
        scores, states_bundle = zip(*results)
        return_ = np.stack(scores, 0).sum(1)
        actions = np.array(actions)[np.argsort(return_)]
        mean, std = actions[-topk:].mean(0), actions[-topk:].std(0)
        # top_actions = np.array(actions)[np.argpartition(scores, -topk)[-topk:]]
        # mean, std = top_actions.mean(0), top_actions.std(0)
    return mean[0]
    # idx = np.argmax(return_)
    # best_a = np.array(actions)[idx][0]
    # print('best_a', best_a)
    # img_r = scores[idx][0]
    # best_state = states_bundle[idx]
    # print('bundle', best_state)
    # # test
    # env = env_ctor()
    # env.reset()
    # with env.physics.reset_context():
    #     env.physics.data.qpos[:] = state[0]
    #     env.physics.data.qvel[:] = state[1]
    # time_step = env.step(best_a)
    # new_state = np.array([env.physics.data.qpos, env.physics.data.qvel])
    # # print(np.all(new_state == best_state))
    # print('test', state,new_state)
    # assert (state == best_state[0]).all(), 'weird begin{}\n {}'.format(new_state, best_state)
    # assert (new_state == best_state[1]).all(), 'weird end{}\n {}'.format(new_state, best_state)
    # assert img_r == time_step.reward, 'different reward'
    # exit()
    # return best_a, img_r, new_state



def get_repeat(domain):
    if domain in ['walker', 'finger']:
        repeat = 2
    elif domain in ['cheetah', 'cup', 'reacher']:
        repeat = 4
    elif domain in ['cartpole']:
        repeat = 8
    else:
        raise NotImplementedError('{} env cannot be found'.format(domain))
    return repeat


def create_env(args):
    repeat = get_repeat(args.domain)
    env = suite.load(args.domain, args.task)
    env = ActionRepeat(env, repeat)
    return env


def main(args, evaluator):
    scores = []
    durations = []
    random = np.random.RandomState(0)
    env_ctor = functools.partial(create_env, args)
    env = env_ctor()
    action_spec = env.action_spec()
    for i in range(args.episodes):

        score = 0
        img_score = 0
        durations.append(0)
        time_step = env.reset()
        while not time_step.last():
            state = np.array([env.physics.data.qpos, env.physics.data.qvel])
            # print('initial', state)
            # print(env.physics.named.data)
            # print(env.physics.named.data.geom_xpos)
            # # print(env.physics.named.data.geom_pos)
            # print(env.physics.named.data.qpos)
            #
            # print(env.physics.named.data.geom_xpos['finger', :2])
            # print(env.physics.named.data.geom_xpos['target', :2])
            # exit()
            action = cem_planner(
                evaluator, random, action_spec, state,
                args.horizon, args.proposals, args.topk, args.iterations, env_ctor)
            # img_score += imag_r
            # if action.max()>1 or action.min()<-1:
            #     counter+=1
            # print(counter, ' action range', action.max(), action.min())
            action = np.clip(action, -1, 1)
            time_step = env.step(action)
            score += time_step.reward
            durations[-1] += 1
            # if not args.quiet:
            print(durations[-1], score, img_score, flush=True)
        scores.append(score)
    durations = np.array(durations)
    scores = np.array(scores)
    if not args.quiet:
        print(durations)
        print(scores)
        print('Mean episode length:', durations.mean())
        print('Mean score:         ', scores.mean())
        print('Standard deviation: ', scores.std())
    else:
        # print(scores.mean())
        pass
    return scores.mean()


if __name__ == '__main__':
    boolean = lambda x: ['False', 'True'].index(x)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'domain', default='cheetah',
        help='Name of the environment to load.')
    parser.add_argument(
        'task', default='run',
        help='Name of the task to load.')
    parser.add_argument(
        '-r', '--repeat', type=int, default=4,
        help='Number of times to repeat each action for.')
    parser.add_argument(
        '-e', '--episodes', type=int, default=5,
        help='Number of episodes to average over.')
    parser.add_argument(
        '-l', '--horizon', type=int, default=12,
        help='Length of each action sequence to consider.')
    parser.add_argument(
        '-p', '--proposals', type=int, default=1000,
        help='Number of action sequences to evaluate per iteration.')
    parser.add_argument(
        '-k', '--topk', type=int, default=100,
        help='Number of best action sequences to refit belief to.')
    parser.add_argument(
        '-i', '--iterations', type=int, default=10,
        help='Number of optimization iterations for each action sequence.')
    args = parser.parse_args()
    args.quiet = False

    env_ctor = functools.partial(create_env, args)
    evaluator = AsyncEvaluator(env_ctor, 40)
    t1 = time.time()
    print('begin')
    score = main(args, evaluator)
    # ls = 6, 8, 10, 12, 14
    # ps = 1000, 500, 300, 100
    # fs = 0.5, 0.3, 0.1, 0.05
    # is_ = 3, 5, 10, 15
    # args.quiet = True
    # with open('mpc.csv', 'w') as outfile:
    #     outfile.write('horizon,proposals,fraction,iterations,score\n')
    #     for l, p, f, i in itertools.product(ls, ps, fs, is_):
    #         print('start')
    #         args.horizon = l
    #         args.proposals = p
    #         args.topk = int(p * f)
    #         args.iterations = i
    #         score = main(args, evaluator)
    #         row = '{},{},{},{},{}\n'.format(l, p, f, i, score)
    #         print(row)
    #         outfile.write(row)
    #         outfile.flush()
    print('TIME EPLAPSED ', time.time() - t1)
    print(score)
    evaluator.close()
