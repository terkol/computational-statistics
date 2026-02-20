import numpy as np

def log_prob(theta):
    return -0.5 * theta @ (Sigma_inv @ theta)

def grad_log_prob(theta):
    return -(Sigma_inv @ theta)

def mh_rw_step(theta, sigma):
    proposal = theta + sigma * np.random.randn(2)
    log_alpha = log_prob(proposal) - log_prob(theta)
    if np.log(np.random.rand()) < log_alpha:
        return proposal
    return theta

def run_mh_many(start_theta, n_steps, n_runs, sigma):
    distances = []
    for _ in range(n_runs):
        theta = start_theta.copy()
        for _ in range(n_steps):
            theta = mh_rw_step(theta, sigma)
        distances.append(np.linalg.norm(theta - start_theta))
    return np.array(distances)

def leapfrog(theta, r, eps, L):
    t, p = theta.copy(), r.copy()
    p += 0.5 * eps * grad_log_prob(t)
    for _ in range(L - 1):
        t += eps * p
        p += eps * grad_log_prob(t)
    t += eps * p
    p += 0.5 * eps * grad_log_prob(t)
    return t, p

def hmc_one_step(theta, eps, L):
    r0 = np.random.randn(2)
    H0 = -log_prob(theta) + 0.5 * (r0 @ r0)
    t_prop, r_prop = leapfrog(theta, r0, eps, L)
    H1 = -log_prob(t_prop) + 0.5 * (r_prop @ r_prop)
    if np.log(np.random.rand()) < -(H1 - H0):
        return t_prop
    return theta

def run_hmc_many(start_theta, eps, L, n_runs):
    distances = []
    for _ in range(n_runs):
        theta1 = hmc_one_step(start_theta, eps, L)
        distances.append(np.linalg.norm(theta1 - start_theta))
    return np.array(distances)

def find_avgs(start_theta):
    d_mh_1 = run_mh_many(start_theta, n_short, runs, sigma_rw)
    d_mh_100 = run_mh_many(start_theta, n_long, runs, sigma_rw)
    avg_mh_1, avg_mh_100 = d_mh_1.mean(), d_mh_100.mean()
    rw_scaling_ratio_mh = avg_mh_100 / (np.sqrt(n_long) * avg_mh_1)

    d_hmc_L1 = run_hmc_many(start_theta, eps_hmc, L=1, n_runs=runs)
    d_hmc_L100 = run_hmc_many(start_theta, eps_hmc, L=100, n_runs=runs)
    avg_hmc_L1, avg_hmc_L100 = d_hmc_L1.mean(), d_hmc_L100.mean()
    rw_scaling_ratio_hmc = avg_hmc_L100 / (np.sqrt(n_long) * avg_hmc_L1)

    print()
    print(f"RW-MH avg distance (1 step):    {avg_mh_1}")
    print(f"RW-MH avg distance (100 steps): {avg_mh_100}")
    print(f"RW-MH ratio vs sqrt(n) scaling: {rw_scaling_ratio_mh}")

    print(f"HMC avg distance (L=1):         {avg_hmc_L1}")
    print(f"HMC avg distance (L=100):       {avg_hmc_L100}")
    print(f"HMC ratio vs sqrt(n) scaling:   {rw_scaling_ratio_hmc}")

if __name__ == "__main__":
    rho = 0.998
    Sigma_inv = (1.0/(1-rho**2)) * np.array([[1.0, -rho], [-rho, 1.0]])
    np.random.seed(42)
    sigma_rw = 0.01
    eps_hmc = 0.01
    runs = 100
    n_short, n_long = 1, 100

    find_avgs(np.array([0.0, 0.0]))
    find_avgs(np.array([-2.0, 2.0]))