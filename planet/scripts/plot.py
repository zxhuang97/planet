import numpy as np
import matplotlib.pyplot as plt
import argparse
import functools
import os
import sys
from scipy import stats

# name = 'hard_negative'
# name = 'contra_traj12'
name = 'contra_step'
OUT_DIR = 'out/' + name

PALETTE = 10 * (
    '#377eb8', '#4daf4a', '#984ea3', '#e41a1c', '#ff7f00', '#a65628',
    '#f781bf', '#888888', '#a6cee3', '#b2df8a', '#cab2d6', '#fb9a99',
    '#fdbf6f')


def binning(xs, ys, bins, reducer):
    binned_xs = np.arange(xs.min(), xs.max() + 1e-10, bins)
    binned_ys = []
    for start, stop in zip([-np.inf] + list(binned_xs), binned_xs):
        left = (xs <= start).sum()
        right = (xs <= stop).sum()
        binned_ys.append(reducer(ys[left:right]))
    binned_ys = np.array(binned_ys)
    return binned_xs, binned_ys


def linregress(ax, x, y, xlabel, ylabel, title=None):
    a, b, correl, pvalue, stderr = stats.linregress(x, y)
    xs = np.linspace(x.min() - 0.1, x.max() + 0.1, 1000)
    ys = a * xs + b
    ax.scatter(x, y, s=0.1, c=PALETTE[2], alpha=0.2)
    ax.plot(xs, ys, c=PALETTE[1], label='fit')
    ax.plot(xs, xs, c=PALETTE[4], label='1:1')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    print('Linear fit of {} and {}'.format(ylabel, xlabel))
    print('y = {}*x + {} , stderr {}, correlation {}'.format(a, b, stderr, correl))


def reward_fit(gd, pred):
    print('#######################')
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    linregress(ax, gd, pred, xlabel='Ground truth reward', ylabel='Predicted reward')

    ax.set_xlim(gd[0] - 0.2, gd[-1] + 0.2)
    ax.set_ylim(pred[0] - 0.2, pred[-1] + 0.2)
    plt.legend()
    plt.savefig(os.path.join(OUT_DIR, 'reward_fit.png'))
    plt.show()


def reward_hist(gd, pred, bins):
    plt.hist(x=gd, bins=bins, rwidth=0.85, alpha=0.7, color=PALETTE[0], label='gt')
    plt.hist(x=pred, bins=bins, rwidth=0.85, alpha=0.7, color=PALETTE[1], label='pred')
    plt.title('Distribution of reward')
    plt.legend()
    plt.savefig(os.path.join(OUT_DIR, 'reward_hist.png'))
    plt.show()


def diff_hist(gd, pred, bins):
    diff = gd - pred
    plt.hist(x=diff, bins=bins, rwidth=0.85, alpha=0.7, color=PALETTE[0], label='Diff')
    plt.title('Distribution of difference(gd-pred)')
    plt.legend()
    plt.savefig(os.path.join(OUT_DIR, 'diff_hist.png'))
    plt.show()


def upper_tri_masking(A):
    m = A.shape[0]
    r = np.arange(m)
    mask = r[:, None] < r
    return A[mask]


def dist_reg(gd, pred, prefix='', hist=False):
    # calculate pair-wise distance
    gd = gd.reshape(1, -1)
    pred = pred.reshape(1, -1)
    all_dist_gd = gd - gd.T
    all_dist_pred = pred - pred.T
    all_dist_gd = upper_tri_masking(all_dist_gd)
    all_dist_pred = upper_tri_masking(all_dist_pred)
    print('num of dist {}'.format(all_dist_pred.shape[0]))
    if hist:
        plt.hist(x=all_dist_gd, bins='auto', rwidth=0.85, alpha=0.7, color=PALETTE[2], label='gt')
        plt.hist(x=all_dist_pred, bins='auto', rwidth=0.85, alpha=0.7, color=PALETTE[1], label='pred')
        plt.title('Distribution of pair-wise distance')
        plt.legend()
        plt.savefig(os.path.join(OUT_DIR, 'pairwise_hist.png'))
        plt.show()

    # sorting
    sorted_dis = np.array([[g, p] for g, p in sorted(zip(all_dist_gd, all_dist_pred))])  # [X,2]
    all_dist_gd = sorted_dis[:, 0].reshape(-1)
    all_dist_pred = sorted_dis[:, 1].reshape(-1)

    # plot configuration
    xlabel = 'Pair-wise distance of gd'
    ylabel = 'Pair-wise distance of pred'
    rows = 1
    cols = 3
    figsize = cols * 5.5, rows * 5
    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    dist_gds, dist_preds = stratify(all_dist_gd, all_dist_pred)
    titles = [' + all margins', ' + smallest 20% margins', ' + largest 20% margins']
    accs = []
    for i, (dist_gd, dist_pred, t) in enumerate(zip(dist_gds, dist_preds, titles)):
        acc = np.mean(dist_gd - dist_pred < 0.0) * 100
        title = prefix + t + ': {}% acc'.format(int(acc))
        accs.append(acc)
        print()
        print('#######################')
        print(title)
        linregress(axes[i], dist_gd, dist_pred, xlabel=xlabel, ylabel=ylabel, title=title)
        axes[i].set_xlim(all_dist_gd[0] - 0.2, all_dist_gd[-1] + 0.2)
        axes[i].set_ylim(all_dist_pred[0] - 0.2, all_dist_pred[-1] + 0.2)
        print('{}% of the paris confomrs with the constraint'.format(acc))

    for ax in axes.flatten():
        ax.set_xlim(all_dist_gd[0] - 0.2, all_dist_gd[-1] + 0.2)
        ax.set_ylim(all_dist_pred[0] - 0.2, all_dist_pred[-1] + 0.2)
    plt.legend()
    plt.savefig(os.path.join(OUT_DIR, 'dist_' + prefix + '.png'))
    plt.show()
    return accs


def stratify(x, y, num_part=5):
    parti_s = x.shape[0] // num_part
    xs, ys = [], []
    xs.append(x)
    ys.append(y)
    xs.append(x[:parti_s])
    ys.append(y[:parti_s])
    xs.append(x[-parti_s:])
    ys.append(y[-parti_s:])
    return xs, ys


def distance_diagnostic(all_gd, all_pred, hor):
    gds, preds = stratify(all_gd, all_pred)
    prefixes = ['all data', 'lower 20% reward ', 'top 20% reward ']
    prefixes = ['hori{} '.format(hor) + p for p in prefixes]
    accs = []
    for i, (gd, pred, pref) in enumerate(zip(gds, preds, prefixes)):
        hist = False if i > 0 else True
        print()
        data_stats(gd, pref + 'Ground truth reward')
        data_stats(pred, pref + 'Predicted reward')
        accs += dist_reg(gd, pred, pref, hist)
    return accs


def horizon_sum(input, horizon=12):
    bs, epi_len = 50, 50
    new_w = epi_len - horizon + 1
    weights = np.zeros([epi_len, new_w])
    for i in range(new_w):
        weights[i:i + horizon, i] = 1.0
    horizon_sum = np.matmul(input, weights)
    return horizon_sum


def reward_diagnostic(rewards, bins=20):
    gd, pred = prepare(rewards, sort=True)
    data_stats(gd, 'Ground truth reward')
    data_stats(pred, 'Predicted reward')
    reward_hist(gd, pred, bins=bins)
    reward_fit(gd, pred)

    partial_rewards = np.random.permutation(rewards)[:2000]
    gd, pred = prepare(partial_rewards, sort=True)
    diff_hist(gd, pred, bins=bins)

    horizons = [1, 4, 8, 12, 16]

    gd, pred = prepare(rewards)
    gd = gd.reshape(-1, 50)
    pred = pred.reshape(-1, 50)
    accs = []
    for hor in horizons:
        hor_gd = horizon_sum(gd, horizon=hor).reshape(-1, 1)
        hor_pred = horizon_sum(pred, horizon=hor).reshape(-1, 1)
        temp_rewards = np.concatenate((hor_gd, hor_pred), 1)
        partial_rewards = np.random.permutation(temp_rewards)[:2000]
        part_hor_gd, part_hor_pred = prepare(partial_rewards, sort=True)
        accs += distance_diagnostic(part_hor_gd, part_hor_pred, hor)
    for a in accs:
        print("%.1f" % a)


def data_stats(data, name):
    print('#######################')
    print(name)
    print('Min: {}    Max: {}    Mean: {}    Std: {}'.format(
        data.min(), data.max(), np.nanmean(data), np.nanstd(data)))


def prepare(rewards, sort=True):
    if sort:
        sorted_r = np.array([[g, e] for g, e in sorted(zip(rewards[:, 0], rewards[:, 1]))])  # [X,2]
        gd = sorted_r[:, 0].reshape(-1)
        pred = sorted_r[:, 1].reshape(-1)
    else:
        gd = rewards[:, 0]
        pred = rewards[:, 1]
    return gd, pred


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--logdir', required=True)
    args = parser.parse_args()
    rewards = np.load(os.path.join(args.logdir, '{}.npy'.format(name)))
    rewards = np.random.permutation(rewards)[:100000]
    if os.path.exists(OUT_DIR) is False:
        os.mkdir(OUT_DIR)

    reward_diagnostic(rewards, bins='auto')
