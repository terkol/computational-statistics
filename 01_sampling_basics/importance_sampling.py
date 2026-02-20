import numpy as np
from numpy.random import default_rng

rng = default_rng(42)

def laplace_sample(n, b):
    u = rng.random(n) - 0.5
    return -b * np.sign(u) * np.log1p(-2 * np.abs(u))

def laplace_logpdf(x, b):
    return -np.abs(x) / b - np.log(2 * b)

def normal_logpdf(x, mu, var):
    return -0.5 * (np.log(2 * np.pi * var) + (x - mu) ** 2 / var)

def mixture_logpdf(x, var):
    a = np.log(0.5) + normal_logpdf(x, -1, var)
    b = np.log(0.5) + normal_logpdf(x,  1, var)
    m = np.maximum(a, b)
    return m + np.log(np.exp(a - m) + np.exp(b - m))

def importance_expectation(var, n_samples=600000, n_repeats=5, b_grid=(0.2, 0.3, 0.5, 0.8, 1.2)):
    pilot = laplace_sample(3000, 1)
    best_ess, best_b = -1, None
    for b in b_grid:
        lw = mixture_logpdf(pilot, var) - laplace_logpdf(pilot, b)
        w = np.exp(lw - lw.max())
        ess = w.sum() ** 2 / (w @ w)
        if ess > best_ess:
            best_ess, best_b = ess, b

    estimates = []
    for _ in range(n_repeats):
        theta = laplace_sample(n_samples, best_b)
        lw = mixture_logpdf(theta, var) - laplace_logpdf(theta, best_b)
        w = np.exp(lw - lw.max())
        estimates.append((w * (theta - 1) ** 2).sum() / w.sum())
    return best_b, float(np.mean(estimates)), float(np.std(estimates, ddof=1))

if __name__ == "__main__":
    b1, mean1, sd1 = importance_expectation(0.5)
    print(f"var=0.5  best_b={b1}  expectation={mean1}  MC_sd={sd1}")

    b2, mean2, sd2 = importance_expectation(0.1)
    print(f"var=0.1  best_b={b2}  expectation={mean2}  MC_sd={sd2}")