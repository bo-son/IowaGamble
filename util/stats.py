import itertools
import numpy as np

from psis import psisloo


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def core_summary(samples, params, *args):
    """
    Prints mean and SDs of parameters.

    Parameter
    --------
    samples: extract object
    params: list of strings

    """
    print('Group parameters of:', ''.join(args))
    for param in params:
        print('{:12s} {: 2.3f} {: 2.3f}'.format(param, np.round(np.mean(samples[param]), 3), np.round(np.std(samples[param]), 3)))
    print('----------------------------')


def loo(data, groups):
    """
    Parameters
    ----------
    dtaa: OrderedDict of samples(extract) from different models
    groups: Groups to compare
    
    """
    print("PSIS leave-one-out cross validation.")
    print("Estimates are unreliable if k > 0.7")
    
    for model, samples in data.items():
        print('\n{:8s}'.format(model))
        
        print('{:2s}  {:9s}'.format('-', ' ELPD'))
        for i, group in enumerate(groups):
            elpd = psisloo(samples[i]['log_lik'])[0]
            print('{:2s}  {: 5.3f}'.format(group, elpd), end="  ")
            unreliable = np.where(psisloo(samples[i]['log_lik'])[2] > 0.7)[0].size
            if unreliable > 0:
                print(f'Note: there are {unreliable} unreliable points.')
    
    print("\n--------------------------\nModel Comparisons: Estimate of ELPD difference")
    pairs = itertools.combinations(data.items(), 2)
    for pair in pairs:
        model0, samples0 = pair[0]
        model1, samples1 = pair[1]
        print('\n{:7s} - {:7s}'.format(model0, model1))
        print('{:2s}  {:8s}  {:5s}'.format('', 'estimate', 'SE'))
        for i, group in enumerate(groups):
            elpd_diff = psisloo(samples0[i]['log_lik'])[0] - psisloo(samples1[i]['log_lik'])[0]
            elpd_diff_pointwise = psisloo(samples0[i]['log_lik'])[1] - psisloo(samples1[i]['log_lik'])[1]
            elpd_diff_se = np.sqrt(len(elpd_diff_pointwise) * np.var(elpd_diff_pointwise))
            print('{:2s}  {: 5.3f}  {:5.3f}'.format(group, elpd_diff, elpd_diff_se))