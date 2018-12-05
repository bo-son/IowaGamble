import os
import pickle
from collections import OrderedDict

import numpy as np
import pandas as pd
from scipy import stats
import pystan as ps
import matplotlib.pyplot as plt
import seaborn as sns

from data import load_data
import plot
import simul
from util.stats import core_summary, loo


def initdict():
    """
    Initialization for wsls_rl
    """
    init = dict(mu_p=[.5]*13,
               sig=[1.]*13,
               alpha_r=[.5]*data['N'],
               gamma_r=[.5]*data['N'],
               ev_init_r=[.5]*data['N'],
               zeta_r=[.5]*data['N'],
               lambda_r=[.5]*data['N'],
               ep_r=[.5]*data['N'],
               ws_init_r=[.5]*data['N'],
               ls_init_r=[.5]*data['N'],
               theta_ws_r=[.5]*data['N'],
               theta_ls_r=[.5]*data['N'],
               ws_fin_r=[.5]*data['N'],
               ls_fin_r=[.5]*data['N'],
               k_r=[.5]*data['N']
              )
    return init


def sampling(model, data, trace_label, extract_label, config):
    """
    Parameters
    ----------
    model : fp of pickled Stan model
    data : data object
    trace_label : savefile name for pickled fit object
    extract_label : savefile name for pickled fit.extract() object
    config : dict
    
    Returns
    -------
    trace : a StanFit4model object.
    
    """
    with open(model, 'rb') as fp:
        model = pickle.load(fp)

    # StanFit4model
    trace = model.sampling(data=data, 
                           iter=config['n_draw'], 
                           warmup=config['n_warmup'],
                           chains=config['n_chain'],
                           n_jobs=config['n_core'],
                           algorithm='NUTS',
                           verbose=True
                           )
    
    samples = trace.extract(permuted=True)
    with open(extract_label, 'wb') as fp:
        pickle.dump(samples, fp)
        
    with open(trace_label, 'wb') as fp:
        pickle.dump(trace, fp)
        
    return trace



def main():
    
    for dr in ['model', 'samples', 'output']:
        if not os.path.exists(dr):
            os.makedirs(dr)

    ##################
    ### LOAD DATA. ###
    ##################

    groups      = ['healthy', 'amphetamine', 'heroin']
    groups_abv  = ['HR', 'AM', 'HR']
    models      = ['vpp', 'wsls_2', 'wsls_6']

    df_healthy      = pd.read_csv('rawData/IGTdata_healthy_control.csv')
    df_amphetamine  = pd.read_csv('rawData/IGTdata_amphetamine.csv')
    df_heroin       = pd.read_csv('rawData/IGTdata_heroin.csv')
    df_dict = {'healthy': df_healthy,
                'amphetamine': df_amphetamine,
                'heroin': df_heroin
                }

    data_dict = {}
    for key, df in df_dict.items():
        data_dict[key] = load_data(df)


    ################
    ### COMPILE. ###
    ################

    vpp_model = ps.StanModel(file='exec/vpp.stan',
                                model_name='vpp_model',
                                verbose=True
                                )
    with open('model/vpp_model.pkl', 'wb') as fp:
        pickle.dump(vpp_model, fp)

    wsls_2_model = ps.StanModel(file='exec/wsls_2.stan',
                            model_name='wsls_2_model',
                            verbose=True
                            )
    with open('model/wsls_2_model.pkl', 'wb') as fp:
        pickle.dump(wsls_2_model, fp)

    wsls_6_model = ps.StanModel(file='exec/wsls_6.stan',
                            model_name='wsls_6_model',
                            verbose=True
                            )
    with open('model/wsls_6_model.pkl', 'wb') as fp:
        pickle.dump(wsls_6_model, fp)

    wsls_rl_model = ps.StanModel(file='exec/wsls_rl.stan',
                            model_name='wsls_rl_model',
                            verbose=True
                            )
    with open('model/wsls_rl_model.pkl', 'wb') as fp:
        pickle.dump(wsls_rl_model, fp)



    #################
    ### SAMPLING. ###
    #################

    # Summaries are stored here
    if not os.path.exists('output/summary'):
        os.makedirs('output/summary')


    for group, data in data_dict.items():
        config = {'n_draw': 3000, 
                'n_warmup': 1000,
                'n_chain': 3,
                'n_core': 3
                }
        trace = sampling(model='model/vpp_model.pkl',
                        data=data,
                        trace_label=f'samples/vpp_trace_{group}.pkl',
                        extract_label=f'samples/vpp_samples_{group}.pkl',
                        config=config
                        )
        # Check Rhat, ESS
        with open(f'output/summary/vpp_{group}.txt', 'w') as fp:
            fp.write(str(trace))

    for group, data in data_dict.items():
        config = {'n_draw': 3000, 
                'n_warmup': 1000,
                'n_chain': 3,
                'n_core': 3
                }
        trace = sampling(model='model/wsls_2_model.pkl',
                        data=data,
                        trace_label=f'samples/wsls_2_trace_{group}.pkl',
                        extract_label=f'samples/wsls_2_samples_{group}.pkl',
                        config=config
                        )
        # Check Rhat, ESS
        with open(f'output/summary/wsls_2_{group}.txt', 'w') as fp:
            fp.write(str(trace))

    for group, data in data_dict.items():
        config = {'n_draw': 3000, 
                'n_warmup': 1000,
                'n_chain': 3,
                'n_core': 3
                }
        trace = sampling(model='model/wsls_6_model.pkl',
                        data=data,
                        trace_label=f'samples/wsls_6_trace_{group}.pkl',
                        extract_label=f'samples/wsls_6_samples_{group}.pkl',
                        config=config
                        )
        # Check Rhat, ESS
        with open(f'output/summary/wsls_6_{group}.txt', 'w') as fp:
            fp.write(str(trace))


    ###############################
    ### ANALYSIS PRELIMINARIES. ###
    ###############################

    # List of POIs
    indiv_param_vpp     = ['alpha', 'lambda', 'phi', 'k', 'ep_pos', 'ep_neg', 'w', 'c']
    indiv_param_wsls_2  = ['ws', 'ls']
    indiv_param_wsls_6  = ['ws_init', 'ls_init', 'theta_ws', 'theta_ls', 'ws_fin', 'ls_fin']
    indiv_param_wsls_rl = ['alpha', 'gamma', 'ev_init', 'zeta', 'lambda', 'ep', 'ws_init', \
                            'ls_init', 'theta_ws', 'theta_ls', 'ws_fin', 'ls_fin', 'k']
    group_param_vpp     = ['mu_' + param for param in indiv_param_vpp] 
    group_param_wsls_2  = ['mu_' + param for param in indiv_param_wsls_2]
    group_param_wsls_6  = ['mu_' + param for param in indiv_param_wsls_6]
    group_param_wsls_rl = ['mu_' + param for param in indiv_param_wsls_rl]
    group_param_all     = ['mu_p', 'sig']


    # Store samples for future use
    vpp_samples = []
    wsls_2_samples = []
    wsls_6_samples = []
    #wsls_rl_samples = []
    for group in groups:
        with open(f'samples/vpp_samples_{group}.pkl', 'rb') as fp:
            vpp_samples.append(pickle.load(fp))
        with open(f'samples/wsls_2_samples_{group}.pkl', 'rb') as fp:
            wsls_2_samples.append(pickle.load(fp))    
        with open(f'samples/wsls_6_samples_{group}.pkl', 'rb') as fp:
            wsls_6_samples.append(pickle.load(fp))      


    ##########################
    ### CHECK CONVERGENCE. ###
    ##########################

    # Traceplots are stored here
    if not os.path.exists('output/traceplot'):
        os.makedirs('output/traceplot')


    # Traceplot
    for group in groups:
        plot.traceplot(model_fp=f'model/vpp_model.pkl',
                        trace_fp=f'samples/vpp_trace_{group}.pkl',
                        params=group_param_vpp,
                        fn=f'output/traceplot/vpp_{group}.pdf')
                       
        plot.traceplot(model_fp=f'model/wsls_2_model.pkl',
                        trace_fp=f'samples/wsls_2_trace_{group}.pkl',
                        params=group_param_wsls_2,
                        fn=f'output/traceplot/wsls_2_{group}.pdf')

        plot.traceplot(model_fp=f'model/wsls_6_model.pkl',
                        trace_fp=f'samples/wsls_6_trace_{group}.pkl',
                        params=group_param_wsls_6,
                        fn=f'output/traceplot/wsls_6_{group}.pdf')


    ###############################
    ### POSTERIOR DISTRIBUTION. ###
    ###############################

    # Posterior plots are stored here
    if not os.path.exists('output/posterior_plot'):
        os.makedirs('output/posterior_plot')


    # Mean and SDs of group mean parameters
    for i, group in enumerate(groups_abv):
        core_summary(vpp_samples[i], group_param_vpp, group)
        core_summary(wsls_2_samples[i], group_param_wsls_2, group)
        core_summary(wsls_6_samples[i], group_param_wsls_6, group)


    # Posterior distribution of group parameters
    for i, group in enumerate(groups):
        plot.posterior_plot(vpp_samples[i], 
                            params=group_param_vpp,
                            xlabel=indiv_param_vpp,
                            fn=f'output/posterior_plot/vpp_{group}.png'
                            )
        plot.posterior_plot(wsls_2_samples[i], 
                            params=group_param_wsls_2,
                            xlabel=indiv_param_wsls_2,
                            fn=f'output/posterior_plot/wsls_2_{group}.png'
                            )
        plot.posterior_plot(wsls_6_samples[i], 
                            params=group_param_wsls_6,
                            xlabel=indiv_param_wsls_6,
                            fn=f'output/posterior_plot/wsls_6_{group}.png'
                            )
        

    # Parameter-wise posterior
    param_posterior_plot(params=group_param_vpp,
                        samples_lst=vpp_samples,
                        xlabel=[f'[{group}] {param}' for param in indiv_param_vpp for group in groups_abv],
                        fn='output/posterior_plot/paramwise_vpp.png'
                        )
    plot.param_posterior_plot(params=group_param_wsls_2,
                        samples_lst=wsls_2_samples,
                        xlabel=[f'[{group}] {param}' for param in indiv_param_wsls_2 for group in groups_abv],
                        fn='output/posterior_plot/paramwise_wsls_2.png'
                        )
    plot.param_posterior_plot(params=group_param_wsls_6,
                        samples_lst=wsls_6_samples,
                        xlabel=[f'[{group}] {param}' for param in indiv_param_wsls_6 for group in groups_abv],
                        fn='output/posterior_plot/paramwise_wsls_6.png'
                        )


    # Joint posterior (WSLS-2)
    plot.joint_posterior_plot(df=pd.DataFrame({'ws': wsls_2_samples[0]['mu_ws'], 'ls': wsls_2_samples[0]['mu_ls']}),
                            params=['ws', 'ls'],
                            col=sns.xkcd_rgb["denim blue"],
                            fn='output/posterior_plot/joint_wsls_2_healthy.png')
    plot.joint_posterior_plot(df=pd.DataFrame({'ws': wsls_2_samples[1]['mu_ws'], 'ls': wsls_2_samples[1]['mu_ls']}),
                            params=['ws', 'ls'],
                            col=sns.xkcd_rgb["medium green"],
                            fn='output/posterior_plot/joint_wsls_2_amphetamine.png')
    plot.joint_posterior_plot(df=pd.DataFrame({'ws': wsls_2_samples[2]['mu_ws'], 'ls': wsls_2_samples[2]['mu_ls']}),
                            params=['ws', 'ls'],
                            col=sns.xkcd_rgb["pale red"],
                            fn='output/posterior_plot/joint_wsls_2_heroin.png')


    # Association plot (WSLS-6, VPP)
    for i, group in enumerate(groups):
        plot.run_corr_plot(samples=vpp_samples[i],
                        params=group_param_vpp,
                        fn=f'output/posterior_plot/correlation_vpp_{group}.png')
        plot.run_corr_plot(samples=wsls_6_samples[i],
                        params=group_param_wsls_6,
                        fn=f'output/posterior_plot/correlation_wsls_6_{group}.png')


    ###################################
    ### POSTERIOR PREDICTIVE CHECK. ###
    ###################################

    if not os.path.exists('output/posterior_predictive'):
        os.makedirs('output/posterior_predictive')

    # Posterior predictive (shows every trial)
    for i, group in enumerate(groups):
        plot.posterior_pred_plot(samples=vpp_samples[i],
                                title=f'Predicted Deck Choice of {groups_abv[i]} [VPP]',
                                fn=f'output/posterior_predictive/vpp_{group}.png',
                                errband=True,
                                save_df=True)
        plot.posterior_pred_plot(samples=wsls_2_samples[i],
                                title=f'Predicted Deck Choice of {groups_abv[i]} [WSLS-2]',
                                fn=f'output/posterior_predictive/wsls_2_{group}.png',
                                errband=True,
                                save_df=True)
        plot.posterior_pred_plot(samples=wsls_6_samples[i],
                                title=f'Predicted Deck Choice of {groups_abv[i]} [WSLS_6]',
                                fn=f'output/posterior_predictive/wsls_6_{group}.png',
                                errband=True,
                                save_df=True)


    # 10-blockwise Posterior predictive
    for i, group in enumerate(groups):
        plot.posterior_pred_10_plot(samples=vpp_samples[i],
                                    title=f'Predicted Deck Choice of {groups_abv[i]} [VPP]',
                                    fn=f'output/posterior_predictive/10block_vpp_{group}.png')
        plot.posterior_pred_10_plot(samples=wsls_2_samples[i],
                                    title=f'Predicted Deck Choice of {groups_abv[i]} [WSLS_2]',
                                    fn=f'output/posterior_predictive/10block_wsls_2_{group}.png')
        plot.posterior_pred_10_plot(samples=wsls_6_samples[i],
                                    title=f'Predicted Deck Choice of {groups_abv[i]} [WSLS-6]',
                                    fn=f'output/posterior_predictive/10block_wsls_6_{group}.png')


    # 10-blockwise ground truth behavior plot
    for i, group in enumerate(groups):
        plot.gt_10_plot(df=df_dict[group],
                        title=f'Deck Choice of {groups_abv[i]}',
                        fn=f'output/posterior_predictive/gt_{group}.png')



    #####################
    ### POST-HOC FIT. ###
    #####################

    # loo: if positive, model 1 is better than model 2
    samples_dict = OrderedDict()
    samples_dict['VPP'] = vpp_samples
    samples_dict['WSLS-2'] = wsls_2_samples
    samples_dict['WSLS-6'] = wsls_6_samples
    loo(samples_dict, groups_abv)



    ###########################
    ### PARAMETER RECOVERY. ###
    ###########################

    for dr in ['simulData', 'output/recovery']:
        if not os.path.exists(dr):
            os.makedirs(dr)

    # Generate outcome of all decks; identical for all subjects
    # shape (4, 2, 60): each sublist(deck) represents (gain, loss) of 60 cards 
    gainloss = [simul.gen_gainloss(nloss=5, net_start=-250, net_increaseby=-150, gain_range=(16, 29)),
                simul.gen_gainloss(nloss=1, net_start=-250, net_increaseby=-150, gain_range=(16, 29)),
                simul.gen_gainloss(nloss=5, net_start=250, net_increaseby=25, gain_range=(8, 17)),
                simul.gen_gainloss(nloss=1, net_start=250, net_increaseby=25, gain_range=(8, 17))
                ]
    gainloss = np.array(gainloss)

    df_simul_dict = {}
    data_simul_dict = {}

    # vpp
    for i, group in enumerate(groups):
        simul.gen_simul_vpp(samples=vpp_samples[i],
                            params=indiv_param_vpp,
                            size=100,
                            gainloss=gainloss,
                            fn=f'simulData/vpp_{group}.csv'
                            )
    df_simul_dict[('vpp', 'healthy')]     = pd.read_csv('simulData/vpp_healthy.csv')
    df_simul_dict[('vpp', 'amphetamine')] = pd.read_csv('simulData/vpp_amphetamine.csv')
    df_simul_dict[('vpp', 'heroin')]      = pd.read_csv('simulData/vpp_heroin.csv')

    # wsls_2
    for i, group in enumerate(groups):
        simul.gen_simul_wsls_2(samples=wsls_2_samples[i],
                                params=indiv_param_wsls_2,
                                size=100,
                                gainloss=gainloss,
                                fn=f'simulData/wsls_2_{group}.csv'
                                )
    df_simul_dict[('wsls_2', 'healthy')]     = pd.read_csv('simulData/wsls_2_healthy.csv')
    df_simul_dict[('wsls_2', 'amphetamine')] = pd.read_csv('simulData/wsls_2_amphetamine.csv')
    df_simul_dict[('wsls_2', 'heroin')]      = pd.read_csv('simulData/wsls_2_heroin.csv')

    # wsls_6
    for i, group in enumerate(groups):
        simul.gen_simul_wsls_6(samples=wsls_6_samples[i],
                                params=indiv_param_wsls_6,
                                size=100,
                                gainloss=gainloss,
                                fn=f'simulData/wsls_6_{group}.csv'
                                )
    df_simul_dict[('wsls_6', 'healthy')]     = pd.read_csv('simulData/wsls_6_healthy.csv')
    df_simul_dict[('wsls_6', 'amphetamine')] = pd.read_csv('simulData/wsls_6_amphetamine.csv')
    df_simul_dict[('wsls_6', 'heroin')]      = pd.read_csv('simulData/wsls_6_heroin.csv')


    # Sample for simulated data of all models
    for key, df in df_simul_dict.items():
        data_simul_dict[key] = load_data(df)

    for (model, group), data in data_simul_dict.items():
        config = {'n_draw': 3000, 
                'n_warmup': 1000,
                'n_chain': 3,
                'n_core': 3
                }
        sampling(model=f'model/{model}_model.pkl',
                data=data,
                trace_label=f'samples/simul_{model}_trace_{group}.pkl',
                extract_label=f'samples/simul_{model}_samples_{group}.pkl',
                config=config
                )


    # Open simulated samples
    simul_vpp_samples = []
    simul_wsls_2_samples = []
    simul_wsls_6_samples = []
    for group in groups:
        with open(f'samples/simul_vpp_samples_{group}.pkl', 'rb') as fp:
            simul_vpp_samples.append(pickle.load(fp))
        with open(f'samples/simul_wsls_2_samples_{group}.pkl', 'rb') as fp:
            simul_wsls_2_samples.append(pickle.load(fp))
        with open(f'samples/simul_wsls_6_samples_{group}.pkl', 'rb') as fp:
            simul_wsls_6_samples.append(pickle.load(fp))
        

    # parameter recovery plot
    plot.recovery_plot(params=indiv_param_vpp,
                        samples=vpp_samples,
                        simul_samples=simul_vpp_samples,
                        fn='output/recovery/vpp.png'
                        )
    plot.recovery_plot(params=indiv_param_wsls_2,
                        samples=wsls_2_samples,
                        simul_samples=simul_wsls_2_samples,
                        fn='output/recovery/wsls_2.png'
                        )
    plot.recovery_plot(params=indiv_param_wsls_6,
                        samples=wsls_6_samples,
                        simul_samples=simul_wsls_6_samples,
                        fn='output/recovery/wsls_6.png'
                        )



if __name__ == "__main__":
    main()