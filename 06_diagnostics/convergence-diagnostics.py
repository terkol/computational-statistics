import numpy as np
import pandas as pd
import os

def rhat(chains):
    n, m = chains.shape
    chain_means = chains.mean(axis=0)
    overall_mean = chain_means.mean()
    B = n * chain_means.var(ddof=1)
    W = chains.var(axis=0, ddof=1).mean()
    var_hat = (n - 1) / n * W + B / n
    return np.sqrt(var_hat / W)

def split_rhat(chains):
    n, m = chains.shape
    half = n // 2
    split = np.vstack([chains[:half, i] for i in range(m)] +
                      [chains[half:2*half, i] for i in range(m)]).T
    return rhat(split)

def ess(x, mean_known=0.0, var_known=0.9185):
    x = np.asarray(x)
    n = x.size
    x_centered = x - mean_known
    gamma = np.array([np.dot(x_centered[:n-k], x_centered[k:]) / n for k in range(n)])
    rho = gamma / var_known
    t = []
    k = 1
    while k + 1 < n:
        s = rho[k] + rho[k+1]
        if s <= 0:
            break
        t.append(s)
        k += 2
    tau = 1 + 2 * np.sum(t)
    return n / tau

if __name__ == "__main__":
    samples = pd.read_csv(os.path.dirname(os.path.dirname(__file__))+'\\data\\raw\\bimod_samples.txt', sep='\t').values

    r = rhat(samples)
    split_rhat_val = split_rhat(samples)
    ess_chain1 = ess(samples[:, 0])

    print(f"R-hat: {r}")
    print(f"Split R-hat: {split_rhat_val}")
    print(f"ESS (chain 1): {ess_chain1}")
