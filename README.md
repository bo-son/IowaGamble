# IowaGamble
Bayesian modeling of the Iowa Gambling Task for substance dependent individuals.



The models are variants of perseverance models. Look for data.pdf for `IGT_perserverance.pdf` for details (background, model descriptions, results)

Codes are written in [Stan](http://mc-stan.org) + [Pystan](https://pystan.readthedocs.io/en/latest/getting_started.html). 



To run: `make`



./exec: Stan models (code)

./model: Pickled Stan models

./util: Statistical operation modules

./rawData: Original data (Ahn et al., 2014) \[[paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4129374/)\] \[[download](https://figshare.com/articles/IGT_raw_data_Ahn_et_al_2014_Frontiers_in_Psychology/1101324)\]

./simulData: Simulated data for parameter recovery

./output: Output, for analysis



./samples: Generated MCMC samples will be stored here (directory will be made automatically)

