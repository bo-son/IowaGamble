import pickle
import math

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Ellipse
import seaborn as sns

from util.hpd import hpd
from data import make_gt_df


def traceplot(model_fp, trace_fp, params, fn):
    palette = sns.color_palette()
    sns.set(font_scale=1)
    sns.set_style("whitegrid")
    
    # Load model before loading trace
    with open(model_fp, 'rb') as fp:
        pickle.load(fp)

    with open(trace_fp, 'rb') as fp:
        trace = pickle.load(fp)

    ms = trace.extract(permuted=False, inc_warmup=True)
    iter_from = trace.sim['warmup']
    iter_range = np.arange(iter_from, ms.shape[0])
    num_pages = math.ceil(len(params) / 4)

    with PdfPages(fn) as pdf:
        for pg in range(num_pages):
            plt.figure()
            for pos in range(4):
                pi = pg * 4 + pos
                if pi >= len(params): break
                plt.subplot(4, 2, 2*pos+1)
                plt.tight_layout()
                [plt.plot(iter_range + 1, ms[iter_range, ci, pi], color=palette[ci]) for ci in range(ms.shape[1])]
                plt.title(params[pi])
                plt.subplot(4, 2, 2*(pos+1))
                plt.tight_layout()
                [sns.kdeplot(ms[iter_range, ci, pi], color=palette[ci]) for ci in range(ms.shape[1])]
                plt.title(params[pi])
            pdf.savefig()
            plt.close()


def param_posterior_plot(params, samples_lst, xlabel, fn, credmass=0.95):
    """
    For group comparison.
    Draw plots of all groups for each parameter.

    Parameters
    ----------
    param : names of poi 
    samples_lst : list of extracts
    xlabel: list
    
    """
    palette = sns.color_palette("Blues")
    sns.set()

    ncol = len(samples_lst)
    nrow = len(params)

    fig, ax = plt.subplots(nrow, ncol, figsize=(4*ncol, 3*nrow), tight_layout=True, sharex='row', sharey='row')
    
    for i, param in enumerate(params):
        for s, samples in enumerate(samples_lst):
            pts = samples[param]
            hdi = np.round(hpd(pts, 1-credmass), 2)
            mean = round(np.mean(pts), 3)

            ax[i, s].hist(pts, bins=50, facecolor=palette[1], edgecolor=palette[0], density=True)
            ax[i, s].set_xlabel(xlabel[i*3 + s], fontsize=12)
            ax[i, s].plot(0, label=f'M {mean}', alpha=0)
            ax[i, s].plot(hdi, [0, 0], label=f'HDI{int(credmass * 100)} [{hdi[0]},{hdi[1]}]', linewidth=6, color='black')
            ax[i, s].legend(loc='upper left', fontsize=10)
            ax[i, s].set_yticks([])

    plt.savefig(fn, dpi=150)



def posterior_plot(samples, params, xlabel, fn, credmass=0.95):
    """
    Parameters
    ----------
    samples : extract
    params : names of poi
    
    """
    palette = sns.color_palette("Blues")
    sns.set()
    
    n_poi = len(params)
    ncol = min(2, n_poi)
    nrow = math.ceil(n_poi / ncol)

    fig, ax = plt.subplots(nrow, ncol, figsize=(4*ncol, 3*nrow), tight_layout=True)
    ax = ax.flatten()
    
    for i, param in enumerate(params):
        pts = samples[param]
        hdi = np.round(hpd(pts, 1-credmass), 2)
        mean = round(np.mean(pts), 3)

        ax[i].hist(pts, bins=50, facecolor=palette[1], edgecolor=palette[0], density=True)
        ax[i].set_xlabel(xlabel[i], fontsize=12)
        ax[i].plot(0, label=f'M {mean}', alpha=0)
        ax[i].plot(hdi, [0, 0], label=f'HDI{int(credmass * 100)} [{hdi[0]},{hdi[1]}]', linewidth=6, color='black')
        ax[i].legend(loc='upper left', fontsize=10)
        ax[i].set_yticks([])

    plt.savefig(fn, dpi=150)


def joint_posterior_plot(df, params, col, fn):
    """
    Draws joint posterior plot for 2 parameters.

    Parameters
    ----------
    df: pandas dataframe holding relevant samples. Parameters should be column labels.
    params: list of length 2
    col: color
    fn: savefile name
    """
    sns.set(style="ticks", font_scale=2)
    joint = sns.jointplot(x=params[0], y=params[1], data=df, kind="reg", size=10, color=col, xlim=(0.1, 0.5), ylim=(0.6, 1.0))
    joint.savefig(fn, dpi=150)


def posterior_pred_plot(samples, title, fn, errband=False, save_df=False, save_df_fn=None):
    """
    errband: CI = 68
    """
    sns.set()

    # Make dataframe
    trialwise = np.swapaxes(samples['y_pred'], 0, 2).astype(np.int8)
    n_trial, n_subj, n_sample = trialwise.shape
    columns = ['subject', 'deck', 'counts']
    decks = [1, 2, 3, 4]
    trialwise_dfs = []
    for trial_idx, trial_dat in enumerate(trialwise):
        dat = np.empty((0, 4, 2), dtype=int)
        for i, subj in enumerate(trial_dat):
            counts = [(subj == deck_idx).sum() for deck_idx in decks]
            mat = np.asarray(list(zip(decks, counts)))
            mat = np.expand_dims(mat, axis=0)
            dat = np.concatenate((dat, mat), axis=0)
        m, n, r = dat.shape # (48, 4, 2)
        out_arr = np.column_stack((np.repeat(np.arange(m), n), dat.reshape(m*n, -1)))
        out_df = pd.DataFrame(out_arr, columns=columns)
        out_df['trial'] = trial_idx
        out_df['proportion'] = round(out_df['counts'] / n_sample, 3)
        trialwise_dfs.append(out_df)
    all_df = pd.concat(trialwise_dfs, axis=0, join='outer', ignore_index=True, copy=False)
    if save_df:
        if save_df_fn is None:
            df_fn = fn[:-4] + '.csv'
        all_df.to_csv(df_fn)
    
    sns.set(font_scale=2)
    sns.set_palette("husl")
    
    fig, ax = plt.subplots(figsize=(30, 15))
    ax.tick_params(labelsize=16)
    ax.set_title(title)
    sns.tsplot(data=all_df, time="trial", unit="subject", condition="deck", value="proportion", legend=True, ax=ax, err_style=None)
    plt.savefig(fn, dpi=300)
    
    if errband:
        fig, ax = plt.subplots(figsize=(30, 15))
        ax.tick_params(labelsize=16)
        ax.set_title(title)
        sns.tsplot(data=all_df, time="trial", unit="subject", condition="deck", value="proportion", legend=True, ax=ax)
        plt.savefig(fn[:-4] + '_errband' + '.png', dpi=300)



def posterior_pred_10_plot(samples, title, fn, errband=False, save_df=False, save_df_fn=None):
    """
    errband: CI = 68
    """
    sns.set()

    # Make dataframe
    subjwise = np.swapaxes(samples['y_pred'], 0, 1).astype(np.int)
    subjwise = np.swapaxes(subjwise, 1, 2) # (N, 100, 6000)
    n_subj, _, n_trial = subjwise.shape
    columns = ['trial-block', 'deck', 'proportion']
    decks = [1, 2, 3, 4]
    subjwise_dfs = []
    for subj_idx, subj_dat in enumerate(subjwise):
        dat = np.empty((0, 4, 2))
        for t in range(10):
            counts = [(subj_dat[:10*(t+1)] == deck_idx).sum() for deck_idx in decks]
            mat = np.asarray(list(zip(decks, counts / sum(counts))))
            mat = np.expand_dims(mat, axis=0)
            dat = np.concatenate((dat, mat), axis=0)
        m, n, r = dat.shape # (10, 4, 2)
        out_arr = np.column_stack((np.repeat(np.arange(m), n), dat.reshape(m*n, -1)))
        out_df = pd.DataFrame(out_arr, columns=columns)
        out_df['subject'] = subj_idx
        out_df['proportion'] = round(out_df['proportion'], 3)
        subjwise_dfs.append(out_df)
        all_df = pd.concat(subjwise_dfs, axis=0, join='outer', ignore_index=True, copy=False)
    all_df[['trial-block', 'deck']] = all_df[['trial-block', 'deck']].astype(int)

    sns.set(font_scale=1.2)
    sns.set_palette("husl")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.tick_params(labelsize=12)
    ax.set_title(title)
    sns.tsplot(data=all_df, time="trial-block", unit="subject", condition="deck", value="proportion", legend=True, ax=ax, interpolate=True)
    plt.savefig(fn, dpi=150)



def gt_10_plot(df, title, fn):
    sns.set(font_scale=1.2)
    sns.set_palette("husl")

    gt_df = make_gt_df(df)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.tick_params(labelsize=12)
    ax.set_title(title)
    sns.tsplot(data=gt_df, time='trial-block', condition='deck', value='proportion', unit='subject', legend=True, ax=ax, err_style=None)
    
    plt.savefig(fn, dpi=150)


def corr_plot(x, y, **kws):
    """
    This code was taken from:
    http://statmodeling.hatenablog.com/entry/pystan-rstanbook-chap5-1 
    
    """
    r, _ = stats.spearmanr(x, y)
    ax = plt.gca()
    ax.axis('off')
    ellcolor = plt.cm.RdBu(0.5*(r+1))
    txtcolor = 'black' if math.fabs(r) < 0.5 else 'white'
    ax.add_artist(Ellipse(xy=[.5, .5], width=math.sqrt(1+r), height=math.sqrt(1-r), angle=45,
        facecolor=ellcolor, edgecolor='none', transform=ax.transAxes))
    ax.text(.5, .5, '{:.0f}'.format(r*100), color=txtcolor, fontsize=28,
        horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)


def run_corr_plot(samples, params, fn, columns=None):
    """
    Parameters
    ----------
    sample: extract
    params: poi
    fn: save file name
    columns: specify order in plot. By default equal to params.
    
    """
    if columns is None:
        columns=params
    df = pd.DataFrame({param: samples[param] for param in params}, columns=columns)

    sns.set(font_scale=1.5)
    g = sns.PairGrid(df)
    g = g.map_lower(sns.kdeplot, cmap='Blues_d')
    g = g.map_diag(sns.distplot, kde=False)
    g = g.map_upper(corr_plot)
    g.fig.subplots_adjust(wspace=0.05, hspace=0.05)
    for ax in g.axes.flatten():
        for t in ax.get_xticklabels():
            _ = t.set(rotation=40)
    g.savefig(fn, dpi=150)


def recovery_plot_new(params, samples, simul_samples, fn):
    """
    Plot for parameter recovery test.
    Combines all groups.

    Parameters
    ----------
    params: list of individual parameters
    samples : list of original samples
    simul_samples : list of simulated samples
    fn : savefile name
    
    """
    sns.set_style("whitegrid")
    
    # Make pandas dataframe of simulated data.
    simul_data = {}
    n_sample = None
    n_subj = None
    for param in params:
        temp = np.concatenate((simul_samples[0][param], simul_samples[1][param], simul_samples[2][param]), axis=1)
        if n_subj is None: 
            n_sample, n_subj = temp.shape
        data[param] = temp.reshape(-1)
    simul_data['trial'] = np.tile(np.arange(1, n_sample+1), n_subj).reshape(-1)
    simul_data['subj'] = np.repeat(np.arange(1, n_subj+1), n_sample).reshape(-1)
    simul_df = pd.DataFrame(data=data)

    # Plot
    fig, ax = plt.subplots(len(params), 1, figsize=(0.3*n_subj, 5*len(params)), tight_layout=True)

    for i, param in enumerate(params):
        # Make pandas dataframe of true parameter values.
        gt_data = {}
        gt = np.concatenate((samples[0][param], samples[1][param], samples[2][param]), axis=1)
        N = gt.shape[1]  # number of subjects in all groups
        gt_data[param] = np.mean(gt, axis=0)
        gt_data['subj'] = np.arange(1, N+1)
        gt_df = pd.DataFrame(data=gt_data).sort_values(by=[param])

        simul_means = []
        for subjID in gt_df['subj']:
            simul_means.append(simul_df.loc[simul_df['subj']==subjID][param].mean())
        simul_means = np.array(simul_means)

        # Violinplot of all groups
        sns.violinplot(x='subj', y=param, data=simul_df, order=gt_df['subj'], color='white', inner='quartile', ax=ax[i])
        # Add means of recovered values
        sns.regplot(x=np.arange(N), y=simul_means, scatter=True, fit_reg=False, marker='o', color='black', ax=ax[i])
        
        # Add true parameter value on plot
        sns.regplot(x=np.arange(N), y=gt_df[param], scatter=True, fit_reg=False, marker='o', color='red', ax=ax[i])
        
        ax[i].set_xlabel('Subject ID', fontsize=16)
        ax[i].set_ylabel(param, fontsize=20)
        
    plt.savefig(fn, dpi=150)


