import numpy as np
import matplotlib.pyplot as plt
import argparse
from planet import tools
import functools
import os
import sys
from scipy import stats
from collections import OrderedDict

# name = 'hard_negative'
# name = 'contra_traj12'
# name = 'contra_step'
name = 'log_likeli'
name = 'planning'
OUT_DIR = 'out_cem/' + name

PALETTE = 10 * (
    '#377eb8', '#4daf4a', '#984ea3', '#e41a1c', '#ff7f00', '#a65628',
    '#f781bf', '#888888', '#a6cee3', '#b2df8a', '#cab2d6', '#fb9a99',
    '#fdbf6f')
'''
python -m planet.scripts.plot_cem --logdir logload/
'''

def upper_tri_masking(A):
    m = A.shape[0]
    r = np.arange(m)
    mask = r[:, None] < r
    return A[mask]


def stratify_by_iter(trajs, part_s=3):
    num_parts = int(np.ceil(10 / part_s))
    num_iters = trajs.shape[0]
    itr = np.arange(num_iters) % 10
    batch = {}
    prev = np.zeros_like(itr)
    for i in range(num_parts):
        select = (itr < (i + 1) * part_s).astype(int)
        pure_select = select - prev
        batch['{}'.format(i * part_s)] = trajs[pure_select.astype(bool)]
        prev = select
    return batch


def horizon_sum(input, horizon=12):
    partial_input = input[:, :, :horizon, :]
    horizon_sum = np.sum(partial_input, axis=2)
    return horizon_sum


def planning_diagnostic(buffer):
    # (N, 1000, 12, 2)
    traj_list = stratify_by_iter(buffer, 1)
    results = tools.nested.map(iter_diagnostic, traj_list)

    merged = {k: np.array([v[k] for v in results.values()])
              for k in results['0'].keys()
              }
    return merged
# def cal_rela(pred, tgt):


def iter_diagnostic(trajs, pref=None, plot=False):
    hor_trajs = horizon_sum(trajs, 12)  # N*1000*2
    hor_gds = hor_trajs[:, :, 0]
    hor_preds = hor_trajs[:, :, 1]
    k = 100
    acc_topk, score_gd, score_pred, mean_ratio, rela_acc, cstr_acc, rank_loss = [0.0] * 7
    corres_pred_ranks = []
    print(trajs.shape, hor_trajs.shape)

    num = trajs.shape[0]
    for i, (gd, pred) in enumerate(zip(hor_gds, hor_preds)):

        gd_ind = np.argsort(gd)
        pred_ind = np.argsort(pred)
        top_gd_ind = gd_ind[-k:]
        top_pred_ind = pred_ind[-k:]
        acc_topk += np.intersect1d(top_gd_ind, top_pred_ind).shape[0] / num  # top 100 indices

        score_gd += gd[top_gd_ind].mean() / num
        score_pred += gd[top_pred_ind].mean() / num
        mean_ratio += 1.0 * score_gd / score_pred / num

        gd, pred = gd[gd_ind], pred[gd_ind]
        gd, pred = gd.reshape(-1, 1), pred.reshape(-1, 1)
        all_dist_gd = gd - gd.T
        all_dist_pred = pred - pred.T
        all_dist_gd = upper_tri_masking(all_dist_gd)
        all_dist_pred = upper_tri_masking(all_dist_pred)

        rela_acc += np.array((all_dist_gd > 0) == (all_dist_pred > 0)).mean() / num
        cstr_acc += np.mean(all_dist_gd - all_dist_pred < 0.0) * 100 / num
        rank_loss += np.maximum(0.0, all_dist_gd - all_dist_pred).mean() / num
        error = np.abs(all_dist_gd<0).sum()

        # print(error)
        assert(error==0)

        pred_rank = np.zeros_like(pred_ind)
        pred_rank[pred_ind] = np.arange(pred_ind.shape[0])
        corres_pred_ranks.append(np.array([pred_rank[j] for j in gd_ind]))
        if i < 2 and plot:
            plt.scatter(np.arange(corres_pred_ranks[-1].shape[0]), corres_pred_ranks[-1])
            plt.savefig(os.path.join(OUT_DIR, pref + '_sample%d_rank.png' % i))
            plt.cla()

    corres_pred_rank = np.stack(corres_pred_ranks).mean(axis=0)

    results = {'acc_topk': acc_topk, 'score_gd': score_gd, 'score_pred': score_pred, 'mean_ratio': mean_ratio,
               'rela_acc': rela_acc, 'cstr_acc': cstr_acc, 'rank_loss': rank_loss,
               # 'corres_pred_rank': corres_pred_rank,
               'traj_return_gd': data_stats(hor_gds, ' Ground truth reward'),
               'traj_return_pred': data_stats(hor_preds, ' Predicted reward')}
    #print(results['traj_return_gd'],results['traj_return_pred'])
    return results


def data_stats(data, name):
    print('#######################')
    print(name)
    print('Min: {}    Max: {}    Mean: {}    Std: {}'.format(
        data.min(), data.max(), np.nanmean(data), np.nanstd(data)))
    return [np.nanmean(data), np.nanstd(data)]


def plot_std(results, metric,hor):
    type = 'hor{}_'.format(hor) + metric
    for i, (name, v) in enumerate(results.items()):
        mean = v[metric][:, 0]
        std = v[metric][:, 1]
        xs = np.arange(mean.shape[0])
        kw = dict(color=PALETTE[i], alpha=0.1, linewidths=0)
        plt.fill_between(xs, mean - std, mean + std, **kw)
        plt.plot(xs, mean, color=PALETTE[i], label=name)
        plt.scatter(xs, mean, color=PALETTE[-i])
    plt.title(type)
    plt.legend()
    plt.savefig(os.path.join(OUT_DIR, type+ '.png'))
    plt.clf()


def plot_line(results, metric, hor):
    type = 'hor{}_'.format(hor) + metric
    for i, (name, v) in enumerate(results.items()):
        ys = v[metric]
        xs = np.arange(ys.shape[0])
        plt.plot(xs, ys, color=PALETTE[i], label=name)
        plt.scatter(xs, ys, color=PALETTE[-i])
    plt.title(type)
    plt.legend()
    plt.savefig(os.path.join(OUT_DIR, type + '.png'))
    plt.clf()


def plot_rank(results, metric):
    for i, (name, v) in enumerate(results.items()):
        ys = v[metric]
        xs = np.arange(ys.shape[0])
        plt.plot(xs, ys, color=PALETTE[i], label=name)
        print(xs.shape,ys.shape)
        plt.scatter(xs, ys, color=PALETTE[-i])
    plt.title(metric)
    plt.legend()
    plt.savefig(os.path.join(OUT_DIR, metric + '.png'))
    plt.clf()


def plot_all(results, hor):
    metrics = next(iter(results.values())).keys()
    for metric in metrics:
        print('Plot ', metric)
        if 'traj' not in metric:
            plot_line(results, metric, hor)
        else:
            plot_std(results, metric, hor)

    plot_std(results, metric, hor)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--logdir', required=True)
    args = parser.parse_args()

    if os.path.exists(OUT_DIR) is False:
        os.mkdir(OUT_DIR)
    # (N, 1000, 12, 2)
    runs = ['baseline', 'hard_negative', 'contra_traj', 'contra_step','resample_traj']
    # runs = ['hard_negative','baseline']
    result = OrderedDict()
    hor = 12
    for r in runs:
        print('Load the trajs of ', r)
        buffer = np.load(os.path.join(args.logdir, r, '001/cem_traj.npy'))
        result[r] = planning_diagnostic(buffer)

    plot_all(result, hor)
    # curve_std([stats_gd[:, 0], stats_pred[:, 0]], [stats_gd[:, 1], stats_pred[:, 1]],
    #           ['Ground truth', 'Prediction'])

    # curve_std(stats_pred[:, 0], stats_pred[:, 1])

    # reward_diagnostic(rewards, bins='auto')
