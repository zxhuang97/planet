from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import csv
import fnmatch
import functools
import multiprocessing.dummy as multiprocessing
import os
import re
import sys
import traceback

# import imageio
import numpy as np
import skimage.io
import tensorflow as tf
from tensorboard.backend.event_processing import (
    plugin_event_multiplexer as event_multiplexer)

import glob
import dowel
from dowel import logger, tabular
import glob
import pandas as pd

lock = multiprocessing.Lock()


def safe_print(*args, **kwargs):
    with lock:
        print(*args, **kwargs)


def create_reader(logdir):
    reader = event_multiplexer.EventMultiplexer()
    reader.AddRun(logdir, 'run')
    reader.Reload()
    return reader


def extract_values(reader, tag):
    events = reader.Tensors('run', tag)
    steps = [event.step for event in events]
    times = [event.wall_time for event in events]
    values = [tf.make_ndarray(event.tensor_proto) for event in events]
    return steps, times, values


def export_scalar(basename, steps, times, values):
    safe_print('Writing', basename + '.csv')
    values = [value.item() for value in values]
    with tf.gfile.Open(basename + '.csv', 'w') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(('wall_time', 'step', 'value'))
        for row in zip(times, steps, values):
            writer.writerow(row)


def export_image(basename, steps, times, values):
    tf.reset_default_graph()
    tf_string = tf.placeholder(tf.string)
    tf_tensor = tf.image.decode_image(tf_string)
    with tf.Session() as sess:
        for step, time_, value in zip(steps, times, values):
            filename = '{}-{}-{}.png'.format(basename, step, time_)
            width, height, string = value[0], value[1], value[2]
            del width
            del height
            tensor = sess.run(tf_tensor, {tf_string: string})
            # imageio.imsave(filename, tensor)
            skimage.io.imsave(filename, tensor)
            filename = '{}-{}-{}.npy'.format(basename, step, time_)
            np.save(filename, tensor)


def process_logdir(logdir, args):
    clean = lambda text: re.sub('[^A-Za-z0-9_]', '_', text)
    basename = os.path.join(args.outdir, clean(logdir))
    name = basename + '*' + clean(args.tags) + '*'
    found = tf.gfile.Glob(name)
    if len(found) > 0:
        safe_print('1Exists', name, found)
        return found[0]
    if len(tf.gfile.Glob(basename + '*')) > 0 and not args.force:
        safe_print('2Exists', logdir, basename)
        return
    try:
        safe_print('Start', logdir)
        reader = create_reader(logdir)

        for tag in reader.Runs()['run']['tensors']:  # tensors -> scalars
            # print(tag, args_.tags, fnmatch.fnmatch(tag, args.tags))
            if fnmatch.fnmatch(tag, args.tags):
                steps, times, values = extract_values(reader, tag)
                filename = '{}___{}'.format(basename, clean(tag))

                export_scalar(filename, steps, times, values)
                return filename+'.csv'
        # for tag in tags['images']:
        #   if fnmatch.fnmatch(tag, args.tags):
        #     steps, times, values = extract_values(reader, tag)
        #     filename = '{}___{}'.format(basename, clean(tag))
        #     export_image(filename, steps, times, values)
        del reader
        safe_print('Done', logdir)
        # return filename
    except Exception:
        safe_print('Exception', logdir)
        safe_print(traceback.print_exc())


def main(args):
    logdirs = glob.glob(args.logdirs)
    print(logdirs)
    # return
    assert logdirs
    tf.gfile.MakeDirs(args.outdir)
    np.random.shuffle(logdirs)
    pool = multiprocessing.Pool(args.workers)
    worker_fn = functools.partial(process_logdir, args=args)
    x = pool.map(worker_fn, logdirs)
    return x

def benchmark(runs, split):
    phase_accum = []
    phase = np.array([10, 50]).astype(int) + 1
    phase = {'100k': 10, '500k': 50}
    bin_s = int(1)
    for pref, pos in phase.items():
        for r in runs:
            df = pd.read_csv(r)
            scores = df['value'].to_numpy()
            phase_accum.append(scores[pos - bin_s:pos])
        mean, std = phase_accum.mean(), phase_accum.std()
        tabular.record('{}Loss_{}'.format(split, pref), mean)
#
# def benchmark(logdirs, outdir, tag):
#
#
#     clean = lambda text: re.sub('[^A-Za-z0-9_]', '_', text)
#     phase = np.array([10, 50]).astype(int) + 1
#     bin_s = int(1)
#
#     init_Str = ''
#     for p in phase:
#         out_str = '{:<4}k steps '.format(p - 1)
#         for logdir in logdirs:
#             runs = glob.glob(logdir)
#             # print(runs)
#             dir_accum = {}
#             phase_accum = []
#             for run in runs:
#                 basename = os.path.join(outdir, clean(run))
#                 filename = '{}___{}.csv'.format(basename, clean(tag))
#                 df = pd.read_csv(filename)
#                 scores = df['value'].to_numpy()
#                 # print(scores[p - bin_s:p])
#                 phase_accum.append(scores[p - bin_s:p])
#             phase_accum = np.stack(phase_accum)
#             mean, std = phase_accum.mean(), phase_accum.std()
#             out_str += '{:>5}+/-{}'.format(int(mean), int(std))
#         print(out_str)


if __name__ == '__main__':
    boolean = lambda x: ['False', 'True'].index(x)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--logdirs', default='benchmark',
        help='glob for log directories to fetch')
    parser.add_argument(
        '--tags', default='trainer*score',
        help='glob for tags to save')
    parser.add_argument(
        '--outdir', default='events',
        help='output directory to store values')
    parser.add_argument(
        '--force', type=boolean, default=False,
        help='overwrite existing files')
    parser.add_argument(
        '--workers', type=int, default=10,
        help='number of worker threads')
    args_, remaining = parser.parse_known_args()

    args_.outdir = os.path.expanduser(args_.outdir)
    remaining.insert(0, sys.argv[0])
    tags = ['*general/objectives/reward']
    logger.add_output(dowel.StdOutput())
    logger.add_output(dowel.CsvOutput('loss.csv'))

    base_dir = 'benchmark'
    phases = ['train', 'test']
    envs = ['cartpole_swingup']
    # methods = ['aug2', 'aug3']
    # methods = ['baseline3', 'resample_traj4', 'resample_traj6', 'aug2', 'aug3', 'aug4', 'aug5']
    methods = ['aug6']
    # envs = ['finger_spin', 'cartpole_swingup', 'reacher_easy', 'cheetah_run', 'walker_walk', 'cup_catch']

    for tag in tags:
        for env in envs:
            for meth in methods:
                tabular.record('Method', meth)
                for phase in phases:
                    args_.logdirs = os.path.expanduser(os.path.join(base_dir, env, meth, '*/', phase))
                    args_.tags = tag
                    res = main(args_)
                    # print('results', res)

                    stones = {'100k': 10, '500k': 50}
                    # stones = {'500k': 50}
                    bin_s = int(5)
                    for pref, pos in stones.items():
                        phase_accum = []
                        for r in res:
                            df = pd.read_csv(r)
                            scores = df['value'].to_numpy()
                            phase_accum.append(scores[pos - bin_s:pos])
                        phase_accum = np.stack(phase_accum)
                        mean, std = phase_accum.mean(), phase_accum.std()
                        tabular.record('{}Loss_{}'.format(phase, pref), mean)
                logger.log(tabular)
                logger.dump_all()
    logger.remove_all()