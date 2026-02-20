import numpy as np
import pandas as pd
import os

def simulate_ar1(ar_coef, noise_sd, length=200, x0=1.0, rng=np.random.default_rng()):
    x = np.empty(length)
    x[0] = x0
    shocks = rng.normal(0.0, noise_sd, size=length - 1)
    for t in range(length - 1):
        x[t + 1] = ar_coef * x[t] + shocks[t]
    return x

def summaries(series):
    n = series.size
    return np.array([np.mean(series**2), np.sum(series[:-1] * series[1:]) / (n - 1)])

def sd_last_values(num_sims=5000, length=200):
    rng = np.random.default_rng()
    last_vals = [simulate_ar1(0.75, 0.2, length, rng=rng)[-1] for _ in range(num_sims)]
    return np.std(last_vals, ddof=1)

def abc_ar1_posterior(data, epsilon=0.2, num_accept=50, batch_size=200):
    rng = np.random.default_rng()
    target_summary = summaries(data)
    accepted = []

    while len(accepted) < num_accept:
        print(len(accepted))
        ar_draws = rng.uniform(0.0, 1.0, size=batch_size)
        noise_sds = rng.gamma(shape=8.0, scale=1/8, size=batch_size)
        for a, s in zip(ar_draws, noise_sds):
            sim_summary = summaries(simulate_ar1(a, s, len(data), rng=rng))
            if np.linalg.norm(sim_summary - target_summary) <= epsilon:
                accepted.append((a, s))
                if len(accepted) == num_accept:
                    break
    return np.array(accepted)

if __name__ == "__main__":  
    df = pd.read_csv(os.path.dirname(os.path.dirname(__file__))+'\\data\\raw\\toydata.txt', header=None)
    data = pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy()
    print(data)

    print("Data summaries:", summaries(data))
    print("SD of last values at a=0.75, sigma=0.2:", sd_last_values())

    posterior_samples = abc_ar1_posterior(data, epsilon=0.5, num_accept=10)
    a_samples, sigma_samples = posterior_samples[:,0], posterior_samples[:,1]

    print("Posterior mean (a, sigma):", np.mean(a_samples), np.mean(sigma_samples))
    print("Posterior sd   (a, sigma):", np.std(a_samples, ddof=1), np.std(sigma_samples, ddof=1))
