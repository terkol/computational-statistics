import numpy as np
import pandas as pd
import scipy.special as scs
import os

def log_pdf_gamma_shape_rate(x, shape, rate):
    return shape*np.log(rate)-scs.gammaln(shape)+(shape-1)*np.log(x)-rate*x

def log_prior_on_logparam(a, shape, rate):
    return shape*np.log(rate)-scs.gammaln(shape)+(shape-1)*a-rate*np.exp(a)+a

def log_posterior_logalpha_logbeta(a, b, x, prior_shape=1, prior_rate=0.5):
    alpha, beta = np.exp(a), np.exp(b)
    return (np.sum(log_pdf_gamma_shape_rate(x, alpha, beta)) + log_prior_on_logparam(a, prior_shape, prior_rate) + log_prior_on_logparam(b, prior_shape, prior_rate))

def metropolis_hastings_rw(logpost, start_ab, prop_cov, n_samples=15000, burn_in=3000, seed=1):
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(prop_cov)
    cur = np.array(start_ab, float)
    cur_lp = logpost(*cur)
    out = np.empty((n_samples, 2))
    acc = 0
    for t in range(n_samples + burn_in):
        cand = cur + L @ rng.normal(size=2)
        cand_lp = logpost(*cand)
        if np.log(rng.random()) < cand_lp - cur_lp:
            cur, cur_lp = cand, cand_lp
            if t >= burn_in:
                acc += 1
        if t >= burn_in:
            out[t - burn_in] = cur
    return out, acc / n_samples

if __name__ == "__main__":
    data1 = pd.read_csv(os.path.dirname(os.path.dirname(__file__))+'\\data\\raw\\toydata.txt', header=None)
    data1 = data1.values
    data1 = np.array(data1[:,1])
    
    start = [0, 0]
    prop_std = np.array([0.15, 0.15])
    prop_cov = np.diag(prop_std**2)

    ab_samples, acc_rate = metropolis_hastings_rw(lambda a,b: log_posterior_logalpha_logbeta(a, b, data1),start, prop_cov)

    alpha_samples = np.exp(ab_samples[:,0])
    beta_samples  = np.exp(ab_samples[:,1])

    alpha_fix = 1.0
    beta_fix = 1.0
    log_likelihood_at_alpha1 = np.sum(log_pdf_gamma_shape_rate(data1, alpha_fix, beta_fix))
    print(f"log_likelihood_at_alpha1_beta{beta_fix} {log_likelihood_at_alpha1}")

    log_prior_at_a1 = log_prior_on_logparam(1.0, shape=1.0, rate=0.5)
    print(f"log_prior_at_a1 {log_prior_at_a1}")

    print(f"acceptance_rate {acc_rate}")
    print(f"alpha_mean {alpha_samples.mean()}, alpha_sd {alpha_samples.std(ddof=1)}")
    print(f"beta_mean {beta_samples.mean()}, beta_sd {beta_samples.std(ddof=1)}")
