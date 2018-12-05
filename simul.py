import random
import numpy as np
import pandas as pd

from stats import softmax

def fixed_sum_randint(n, total):
    """
    Returns a random list of n positive integers, with fixed sum equal to `total`.
    """
    dividers = sorted(random.sample(range(1, total), n - 1))
    return np.array([a - b for a, b in zip(dividers + [total], [0] + dividers)], dtype=int)


def gen_gainloss(nloss, net_start, net_increaseby, gain_range):
    """
    Parameters
    ----------
    nloss: number of losses out of 10 trials.
    net_start: net outcome of the first block.
    net_increaseby: increase of net outcome by each block.
    gain_range: tuple of (low, high) divided by 5 (to be products of 5)

    Returns
    -------
    (gain, loss): Generated list of gains and losses of each deck. Each list is of length 60, representing 60 cards.

    """
    loss = np.zeros((6, 10), dtype=int)
    gain = np.random.randint(low=gain_range[0], high=gain_range[1], size=(6, 10)) * 5
    for i in range(6):
        # generate block loss
        total = -net_start+(i * -net_increaseby) + np.sum(gain[i])
        rloss = np.append(-fixed_sum_randint(nloss, total), np.zeros(10-nloss, dtype=int))
        np.random.shuffle(rloss)
        loss[i] = rloss
    return gain.reshape(-1), loss.reshape(-1)


def gen_simul_vpp(samples, params, size, gainloss, fn):
    """
    Generates simulated data with 'true' parameter values based on individual parameter estimates.
    For parameter recovery test.

    Parameter
    ---------
    samples: extract
    params: indiv poi (not mu_)
    size: number of trials to generate per subject
    gainloss: (4, 2, 60)
    fn: filename to save

    Returns
    -------
    data: pandas df

    """

    # number of subjects
    N = samples[params[0]].shape[1]

    # Generate a true parameter for each subject based on individual parameter estimates
    true_params = {}
    for param in params:
        true_params[param] = np.mean(samples[param][:, ], axis=0)

    deck = [0, 1, 2, 3]
    theta = np.power(3, true_params['c']) - 1
    ev = np.zeros((N, 4))
    pers = np.zeros((N, 4))
    val = np.zeros((N, 4))

    cnt_by_deck = np.zeros((N, 4), dtype=int) # counter for each deck
    choice_stored = np.zeros((N, size), dtype=int) # choice of each trial
    gain_stored = np.zeros((N, size))
    loss_stored = np.zeros((N, size))
    
    # negative infinity
    nginf = np.finfo(val.dtype).min

    # shape (4, 60): summed gain and loss
    outcome = np.apply_along_axis(np.sum, axis=1, arr=gainloss)

    # vectorized for all subjects
    for t in range(size):
        choice = np.array([np.random.choice(deck, p=softmax(theta[n] * val[n])) for n in range(N)])
        cnt = cnt_by_deck[np.arange(N), choice]

        for n in range(N):
            if cnt[n] >= 60:  # forced to pick from another deck
                org = choice[n]
                temp = theta[n] * val[n]
                temp[org] = nginf
                while 1:
                    choice[n] = np.random.choice(deck, p=softmax(temp))
                    if org != choice[n]: break
                cnt[n] = cnt_by_deck[n, choice[n]]
                
        choice_stored[:, t] = choice + 1    # write [0,1,2,3] as [1,2,3,4]
        gain_stored[:, t] = gainloss[choice, 0, cnt]
        loss_stored[:, t] = gainloss[choice, 1, cnt]
 

        pers = pers * np.repeat(true_params['k'], 4).reshape((N, 4))
        
        is_win = outcome[choice, cnt] >= 0
        util = np.where(is_win,
                        np.power(outcome[choice, cnt], true_params['alpha']),
                        -true_params['lambda'] * np.power(-outcome[choice, cnt], true_params['alpha'])
        )
        pers[np.arange(N), choice] += np.where(is_win, true_params['ep_pos'], true_params['ep_neg'])

        ev[np.arange(N), choice] += true_params['phi'] * (util - ev[np.arange(N), choice])
        val = np.repeat(true_params['w'], 4).reshape((N, 4)) * ev + np.repeat(1 - true_params['w'], 4).reshape((N, 4)) * pers

        cnt_by_deck[np.arange(N), choice] += 1  # increase counter of chosen deck

    data = {'subjID': np.repeat(range(1, N+1), size),
            'trial': np.tile(range(1, 101), N),
            'deck': choice_stored.reshape(-1),
            'gain': gain_stored.reshape(-1),
            'loss': loss_stored.reshape(-1)
            }
    
    df = pd.DataFrame(data=data)
    df.to_csv(fn, index=False)




def gen_simul_wsls_2(samples, params, size, gainloss, fn):
    """
    Generates simulated data with 'true' parameter values based on individual parameter estimates.
    For parameter recovery test.

    Parameter
    ---------
    samples: extract
    params: indiv poi (not mu_)
    size: number of trials to generate per subject
    gainloss: (4, 2, 60)
    fn: filename to save

    Returns
    -------
    data: pandas df

    """

    # number of subjects
    N = samples[params[0]].shape[1]

    # Generate a true parameter for each subject based on individual parameter estimates
    true_params = {}
    for param in params:
        true_params[param] = np.mean(samples[param][:, ], axis=0)

    deck = [0, 1, 2, 3]
    p  = np.full((N, 4), 0.25)     # probability of choosing each option
    prev_outcome = np.zeros(N)     # previous outcome
    cnt_by_deck = np.zeros((N, 4), dtype=int) # counter for each deck
    choice_stored = np.zeros((N, size), dtype=int) # choice of each trial
    gain_stored = np.zeros((N, size))
    loss_stored = np.zeros((N, size))

    # shape (4, 60): summed gain and loss
    outcome = np.apply_along_axis(np.sum, axis=1, arr=gainloss)

    # vectorized for all subjects
    for t in range(size):
        choice = np.array([np.random.choice(deck, p=p[n]) for n in range(N)])
        cnt = cnt_by_deck[np.arange(N), choice]

        choice_stored[:, t] = choice + 1    # write [0,1,2,3] as [1,2,3,4]
        gain_stored[:, t] = gainloss[choice, 0, cnt]
        loss_stored[:, t] = gainloss[choice, 1, cnt]

        is_win = outcome[choice, cnt] >= prev_outcome  # win trial
        p = np.repeat(np.where(is_win, (1 - true_params['ws'])/3, true_params['ls']/3), 4).reshape((N, 4))
        p[np.arange(N), choice] = np.where(is_win, true_params['ws'], 1 - true_params['ls'])

        cnt_by_deck[np.arange(N), choice] += 1  # increase counter of chosen deck
        prev_outcome = outcome[choice, cnt]

    data = {'subjID': np.repeat(range(1, N+1), size),
            'trial': np.tile(range(1, 101), N),
            'deck': choice_stored.reshape(-1),
            'gain': gain_stored.reshape(-1),
            'loss': loss_stored.reshape(-1)
            }
    
    df = pd.DataFrame(data=data)
    df.to_csv(fn, index=False)



def gen_simul_wsls_6(samples, params, size, gainloss, fn):
    """
    Generates simulated data with 'true' parameter values based on individual parameter estimates.
    For parameter recovery test.

    Parameter
    ---------
    samples: extract
    params: indiv poi (not mu_)
    size: number of trials to generate per subject
    gainloss: (4, 2, 60)
    fn: filename to save

    Returns
    -------
    data: pandas df

    """

    # number of subjects
    N = samples[params[0]].shape[1]

    # Generate a true parameter for each subject based on individual parameter estimates
    true_params = {}
    for param in params:
        true_params[param] = np.mean(samples[param][:, ], axis=0)

    deck = [0, 1, 2, 3]
    ws = true_params['ws_init']    # P(stay|win)
    ls = true_params['ls_init']    # P(shift|loss)
    p  = np.full((N, 4), 0.25)     # probability of choosing each option
    prev_outcome = np.zeros(N)     # previous outcome
    cnt_by_deck = np.zeros((N, 4), dtype=int) # counter for each deck
    choice_stored = np.zeros((N, size), dtype=int) # choice of each trial
    gain_stored = np.zeros((N, size))
    loss_stored = np.zeros((N, size))

    # shape (4, 60): summed gain and loss
    outcome = np.apply_along_axis(np.sum, axis=1, arr=gainloss)

    # vectorized for all subjects
    for t in range(size):
        choice = np.array([np.random.choice(deck, p=p[n]) for n in range(N)])
        cnt = cnt_by_deck[np.arange(N), choice]

        choice_stored[:, t] = choice + 1    # write [0,1,2,3] as [1,2,3,4]
        gain_stored[:, t] = gainloss[choice, 0, cnt]
        loss_stored[:, t] = gainloss[choice, 1, cnt]

        
        is_win = outcome[choice, cnt] >= prev_outcome  # win trial
        ws = np.where(is_win,
                      ws + true_params['theta_ws'] * (true_params['ws_fin'] - ws),
                      ws)
        ls = np.where(is_win,
                      ls,
                      ls + true_params['theta_ls'] * (true_params['ls_fin'] - ls))
        p = np.repeat(np.where(is_win, (1-ws)/3, ls/3), 4).reshape((N, 4))
        p[np.arange(N), choice] = np.where(is_win, ws, 1 - ls)


        cnt_by_deck[np.arange(N), choice] += 1  # increase counter of chosen deck
        prev_outcome = outcome[choice, cnt]

    data = {'subjID': np.repeat(range(1, N+1), size),
            'trial': np.tile(range(1, 101), N),
            'deck': choice_stored.reshape(-1),
            'gain': gain_stored.reshape(-1),
            'loss': loss_stored.reshape(-1)
            }
    
    df = pd.DataFrame(data=data)
    df.to_csv(fn, index=False)



