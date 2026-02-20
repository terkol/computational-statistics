import numpy as np
import matplotlib.pyplot as plt

def leapfrog(theta, r, eps, n_steps, cov_inv):
    t, p = theta.copy(), r.copy()
    for _ in range(n_steps):
        p -= 0.5 * eps * (cov_inv @ t)
        t += eps * p
        p -= 0.5 * eps * (cov_inv @ t)
    return t, p

def hamiltonian(theta, r, cov_inv):
    return 0.5 * theta @ (cov_inv @ theta) + 0.5 * (r @ r)

rho = 0.998
cov_inv = (1.0/(1-rho**2)) * np.array([[1.0, -rho], [-rho, 1.0]])
theta0 = np.array([0.0, 0.0])
r0 = np.array([1.0, 1.0/3.0])
L = 10
eps_grid = np.round(np.arange(0.001, 0.100 + 0.001, 0.001), 3)
H0 = hamiltonian(theta0, r0, cov_inv)

accept_prob = []
for eps in eps_grid:
    theta_L, r_L = leapfrog(theta0, r0, eps, L, cov_inv)
    dH = hamiltonian(theta_L, r_L, cov_inv) - H0
    accept_prob.append(min(1.0, np.exp(-dH)))
accept_prob = np.array(accept_prob)

eps_smallest_below_60 = eps_grid[np.where(accept_prob < 0.60)[0][0]]
eps_largest_above_10 = eps_grid[np.where(accept_prob > 0.10)[0][-1]]

eps = 0.05
n_steps = 500
theta_path = [theta0.copy()]
theta, r = theta0.copy(), r0.copy()
for _ in range(n_steps):
    r -= 0.5 * eps * (cov_inv @ theta)
    theta += eps * r
    r -= 0.5 * eps * (cov_inv @ theta)
    theta_path.append(theta.copy())
theta_path = np.array(theta_path)
dist = np.linalg.norm(theta_path, axis=1)

first_local_max_step = next(k for k in range(1, len(dist)-1) if dist[k] > dist[k-1] and dist[k] > dist[k+1])
first_local_min_step = next(k for k in range(first_local_max_step+1, len(dist)-1) if dist[k] < dist[k-1] and dist[k] < dist[k+1])


if __name__ == "__main__":
    plt.figure()
    plt.plot(eps_grid, accept_prob)
    plt.xlabel('epsilon')
    plt.ylabel('acceptance probability')
    plt.grid()

    plt.figure()
    plt.plot(theta_path[:,0], theta_path[:,1])
    plt.scatter(theta_path[0,0], theta_path[0,1])
    plt.xlabel('theta_1')
    plt.ylabel('theta_2')
    plt.axis('equal')
    plt.grid()

    plt.figure()
    plt.plot(np.arange(len(dist)), dist)
    plt.xlabel('step')
    plt.ylabel('distance from origin')
    plt.grid()
    plt.show()

    print("Smallest eps with acceptance < 0.60:", float(eps_smallest_below_60))
    print("Largest eps with acceptance > 0.10:", float(eps_largest_above_10))
    print("First max at step", int(first_local_max_step))
    print("First min at step", int(first_local_min_step))
