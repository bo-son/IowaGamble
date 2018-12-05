import numpy as np
import pandas as pd


def load_data(df):
    """
    Parameters
    ----------
    df: pandas dataframe of the raw data.
    
    Returns
    -------
    data: data object.
        'N': number of subjects
        'T': maximum number of trials
        'T_subj': np 1d array. [n] number of trials of each subject
        'outcome': np 2d array. [n, t] outcome (gain+loss) at each trial t, for each subject
        'choice': np 2d array. [n, t] deck choice at each trial t, of each subject
    """

    subj_lst = df['subjID'].unique().tolist()
    N = len(subj_lst)   # number of subjects
    T_subj = np.zeros(N, dtype=int)
    for i, subj in enumerate(subj_lst):
        df_subj = df.loc[df['subjID'] == subj]
        T_subj[i] = len(df_subj)
    T = max(T_subj)     # max number of trials

    scale = 100

    outcome = np.zeros((N, T))
    choice = np.zeros((N, T), dtype=int)
    for i, subj in enumerate(subj_lst):
        df_subj = df.loc[df['subjID'] == subj]
        _outcome = (df_subj['gain'] + df_subj['loss']) / scale
        outcome[i, :len(_outcome)] = _outcome
        _choice = df_subj['deck']
        choice[i, :len(_choice)] = _choice

    data = {'N': N,
            'T': T,
            'T_subj': T_subj,
            'outcome': outcome,
            'choice': choice
            }

    return data


def make_gt_df(df):
    """
    block of 10 trials

    """

    decks = [1,2,3,4]
    columns = ['trial-block', 'deck', 'counts']
    gt = df.sort_values('trial', axis=0)
    max_trial_block = math.ceil(gt.iloc[-1]['trial'] / 10)
    n_subj_trial = []

    dat = np.empty((0, 4, 2), dtype=int)
    for t in range(max_trial_block):
        isin = np.array(range(10*t+1, 10*(t+1)))
        mask = np.isin(gt['trial'], isin)
        trial_dat = gt.loc[mask]
        n_subj_trial.extend([len(trial_dat)]*4)
        counts = [(trial_dat['deck'] == deck_idx).sum() for deck_idx in decks]
        mat = np.asarray(list(zip(decks, counts)))
        mat = np.expand_dims(mat, axis=0)
        dat = np.concatenate((dat, mat), axis=0)
    m, n, r = dat.shape
    out_arr = np.column_stack((np.repeat(np.arange(m), n), dat.reshape(m*n, -1)))
    out_df = pd.DataFrame(out_arr, columns=columns)
    out_df['proportion'] = round(out_df['counts'] / n_subj_trial, 3)
    out_df['subject'] = 0
    
    return out_df