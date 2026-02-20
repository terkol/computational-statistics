import numpy as np
import pandas as pd
from scipy.stats import t, norm
import os

def logpost_normal(mu):
    return t.logpdf(observations, df=nu_df, loc=mu, scale=1).sum() + norm.logpdf(mu, 0.0, 1.0)

def logpost_uniform(mu, lower=-5.0, upper=5.0):
    if (mu < lower) or (mu > upper):
        return -np.inf
    return t.logpdf(observations, df=nu_df, loc=mu, scale=1).sum()  

def mh(logpost_fn, start_mu, n_draws=30000, burn_in=5000, thin=1, proposal_sd=0.2, seed=0):
    rng = np.random.default_rng(seed)
    chain = np.empty(n_draws)
    mu = float(start_mu)
    logp = logpost_fn(mu)
    accepts = 0
    for i in range(n_draws):
        mu_prop = mu + rng.normal(0.0, proposal_sd)
        logp_prop = logpost_fn(mu_prop)
        if np.log(rng.random()) < (logp_prop - logp):
            mu, logp = mu_prop, logp_prop
            accepts += 1
        chain[i] = mu
    kept = chain[burn_in::thin]
    return kept, accepts / n_draws

if __name__ == "__main__":
    data = pd.read_csv(os.path.dirname(os.path.dirname(__file__))+'\\data\\raw\\toydata2.txt', sep='\t', header=None)
    
    observations = np.asarray(data) 
    nu_df = 5  
    start_mu = observations.mean()
    samples_normal, acc_normal = mh(logpost_normal, start_mu)
    samples_uniform, acc_uniform = mh(logpost_uniform, start_mu)

    print("Normal prior:   mean =", samples_normal.mean(), "sd =", samples_normal.std(ddof=1), "accept =", acc_normal)
    print("Uniform[-5,5]:  mean =", samples_uniform.mean(), "sd =", samples_uniform.std(ddof=1), "accept =", acc_uniform)