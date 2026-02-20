import numpy as np

def log_unnormalized_density(theta):
    c = np.cos(theta)
    if c == 0:
        return -np.inf  
    else:
        return 2*np.log(abs(c)) - abs(theta)**3

def metropolis_hastings(n, proposal_std, init_theta=0.0):
    rng = np.random.default_rng()
    chain = np.empty(n, dtype=float)
    chain[0] = init_theta
    logp_curr = log_unnormalized_density(init_theta)
    num_accepts = 0
    for i in range(1, n):
        candidate = chain[i - 1] + rng.normal(0.0, proposal_std)
        logp_cand = log_unnormalized_density(candidate)
        if np.log(rng.random()) < (logp_cand - logp_curr):
            chain[i] = candidate
            logp_curr = logp_cand
            num_accepts += 1
        else:
            chain[i] = chain[i - 1]
    return chain, num_accepts / (n - 1)

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    _, acc = metropolis_hastings(10000, 0.5)
    print(f"acceptance with proposal_std=0.5: {acc}")

    grid = np.linspace(0, 2, 101)
    print(grid)
    acc_grid = [metropolis_hastings(10000, s)[1] for s in grid]
    idx = np.argmin(np.abs(np.array(acc_grid)-0.5))
    proposal_std_opt = float(grid[idx])
    acc_opt = float(acc_grid[idx])
    print(f"tuned proposal_std: {proposal_std_opt}  acceptance≈{acc_opt}")

    N, burn = 200000, 20000
    samples, _ = metropolis_hastings(N, proposal_std_opt)
    print(f"E[cos θ] ≈ {np.mean(np.cos(samples[burn:]))}")
