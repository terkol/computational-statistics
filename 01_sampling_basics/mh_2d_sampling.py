import numpy as np

def log_target_ring(theta):
    radius = np.sqrt(np.sum((theta - mu)**2))
    return -3.0 * (radius - r)**2

def log_target_ring_with_l1(theta):
    radius = np.sqrt(np.sum((theta - mu)**2))
    return -3.0 * (radius - r)**2 - np.abs(2*theta[0] - theta[1])

def metropolis_hastings(log_pdf, start, n=100000, burn=10000, proposal_std=(0.6, 0.6)):
    dim = len(start)
    x = np.array(start)
    logp_x = log_pdf(x)
    chain = np.empty((n, dim))
    acc = 0
    std = np.array(proposal_std)
    for i in range(n):
        x_prop = x + rng.normal(scale=std, size=dim)
        logp_prop = log_pdf(x_prop)
        if np.log(rng.random()) < (logp_prop-logp_x):
            x, logp_x = x_prop, logp_prop
            acc += 1
        chain[i] = x
    return chain[burn:], acc/n

def estimate_e_theta_sq(log_pdf, **mh_kwargs):
    samples, acc_rate = metropolis_hastings(log_pdf, start=[0.0, 0.0], **mh_kwargs)
    vals = np.sum(samples**2, axis=1)
    mean = vals.mean()
    return mean, acc_rate


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    mu = np.array([1, 2])
    r = np.sqrt(2)
    mean1, acc1 = estimate_e_theta_sq(log_target_ring, proposal_std=(0.83, 0.83))
    mean2, acc2 = estimate_e_theta_sq(log_target_ring_with_l1, proposal_std=(0.67, 0.67))

    print(f"1: E[θ1^2+θ2^2] ≈ {mean1:.3f}, acceptance {acc1}")
    print(f"2: E[θ1^2+θ2^2] ≈ {mean2:.3f}, acceptance {acc2}")