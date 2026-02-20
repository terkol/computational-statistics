import numpy as np
import numpy.random as npr
import pandas as pd
import matplotlib.pyplot as plt
import scipy.special as scs
import os

def softplus(a):
    return np.log1p(np.exp(-np.abs(a)))+np.maximum(a,0)

def log_likelihood(intercept, coefficients):
    linear_predictor = intercept+p4_x @ coefficients
    return np.sum(p4_y*(-softplus(-linear_predictor))+(1-p4_y)*(-softplus(linear_predictor)))

def log_prior(log_sigma_beta, betas_with_intercept):
    sigma_beta = np.exp(log_sigma_beta)
    nu = 3
    student_t = np.sum(scs.gammaln((nu+1)/2)-scs.gammaln(nu/2)-0.5*np.log(nu*np.pi)-np.log(sigma_beta)-0.5*(nu+1)*np.log1p((betas_with_intercept/sigma_beta)**2/nu))
    shape, rate = 2, 0.1
    gamma_on_sigma = shape*np.log(rate)-scs.gammaln(shape)+shape*log_sigma_beta-rate*sigma_beta
    return student_t+gamma_on_sigma

def log_posterior(theta):
    log_sigma_beta, intercept = theta[0], theta[1]
    coefficients = theta[2:]
    return log_likelihood(intercept, coefficients)+log_prior(log_sigma_beta, theta[1:])

def run_random_walk_mh(total_iters, burn_in, step_scale_init, seed):
    npr.seed(seed)
    current_theta = np.zeros(1+1+num_features)
    step_scale = step_scale_init
    accept_count = 0
    window_size = 100
    retained_samples = []
    current_lp = log_posterior(current_theta)
    for t in range(total_iters):
        proposal = current_theta+step_scale*npr.randn(current_theta.size)
        proposal_lp = log_posterior(proposal)
        if np.log(npr.rand()) < proposal_lp-current_lp:
            current_theta = proposal
            current_lp = proposal_lp
            accept_count += 1
        if t < burn_in and (t+1) % window_size == 0:
            accept_rate = accept_count/window_size
            step_scale *= np.exp(0.1*(accept_rate-0.3))
            accept_count = 0
        if t >= burn_in:
            retained_samples.append(current_theta.copy())
    return np.array(retained_samples)

def split_rhat(chains):
    min_len = min(c.shape[0] for c in chains)
    halves = [np.array_split(c[:min_len], 2) for c in chains]
    segments = np.concatenate(halves,0)
    n = segments.shape[1]
    seg_means = segments.mean(1)
    grand_mean = seg_means.mean(0)
    B = n*((seg_means-grand_mean).var(0, ddof=1))
    W = segments.var(1, ddof=1).mean(0)
    var_hat = (n-1)/n*W+B/n
    return np.sqrt(var_hat/W)


if __name__ == "__main__":
    heart_data = pd.read_csv(os.path.dirname(os.path.dirname(__file__))+'\\data\\raw\\heart_data.txt', sep="\t")
    p4_x = heart_data[["SBP", "DBP", "CIG", "AGE"]].values.astype(float)
    p4_y = heart_data["HAS_CHD"].values.astype(float)
    p4_x -= np.mean(p4_x, 0)
    p4_x /= 2 * np.std(p4_x, 0)
    num_obs, num_features = p4_x.shape
    total_iters, burn_in = 30000, 5000
    chain1 = run_random_walk_mh(total_iters, burn_in, step_scale_init=0.2, seed=1)
    chain2 = run_random_walk_mh(total_iters, burn_in, step_scale_init=0.2, seed=2)
    posterior_log_param = np.vstack([chain1, chain2])

    posterior = posterior_log_param.copy()
    posterior[:,0] = np.exp(posterior[:,0])

    param_names = ["sigma_beta","beta0","beta1","beta2","beta3","beta4"]
    quantiles = np.percentile(posterior, [25,50,75], axis=0)
    summary = pd.DataFrame({"parameter":param_names,"q25":quantiles[0], "median":quantiles[1], "q75":quantiles[2]})
    print(summary)

    rhats = split_rhat([chain1,chain2])
    rhatdf = pd.DataFrame({"parameter":["log_sigma_beta"]+param_names[1:],"R_hat":rhats})
    print(rhatdf)

    corr = np.corrcoef(posterior.T)
    strong = []
    for i in range(len(param_names)):
        for j in range(i+1, len(param_names)):
            if abs(corr[i, j]) > 0.5:
                strong.append((param_names[i], param_names[j], corr[i,j]))
    print("Correlations > 0.5:", [(a,b,float(r)) for a,b,r in strong])

    idx = npr.choice(posterior.shape[0], size=1000, replace=False)
    pair_samples = posterior[idx]
    for i in range(len(param_names)):
        for j in range(i+1,len(param_names)):
            plt.figure()
            plt.scatter(pair_samples[:,i], pair_samples[:,j], s=8, alpha=0.5)
            plt.xlabel(param_names[i])
            plt.ylabel(param_names[j])
            plt.title(f"{param_names[i]} vs {param_names[j]}")
            plt.tight_layout()
    plt.show()
