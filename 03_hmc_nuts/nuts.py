import numpy as np

def log_prob(theta): 
    return -0.5 * theta @ (Sigma_inv @ theta)

def grad_log_prob(theta): 
    return -(Sigma_inv @ theta)

def hamiltonian(theta, r): 
    return -log_prob(theta) + 0.5*(r@r)

def run_simplified_nuts(theta0, r0, eps):
    theta_path, r_path = [theta0.copy()], [r0.copy()]
    theta_curr, r_curr = theta0.copy(), r0.copy()
    j = 1
    while True:
        L = 2**(j-1)
        seg_thetas, seg_rs = [], []
        t, p = theta_curr.copy(), r_curr.copy()
        for _ in range(L):
            p += 0.5*eps*grad_log_prob(t)
            t += eps*p
            p += 0.5*eps*grad_log_prob(t)
            seg_thetas.append(t.copy()) 
            seg_rs.append(p.copy())
        theta_path += seg_thetas; r_path += seg_rs
        theta_minus, r_minus = seg_thetas[0], seg_rs[0]
        theta_plus,  r_plus  = seg_thetas[-1], seg_rs[-1]
        delta_theta = theta_plus - theta_minus
        if (delta_theta @ r_minus) < 0 or (delta_theta @ r_plus) < 0:
            return j, theta_plus, r_plus, theta_path, r_path
        theta_curr, r_curr = theta_plus, r_plus
        j += 1

rho = 0.998
Sigma_inv = (1.0/(1-rho**2)) * np.array([[1.0, -rho], [-rho, 1.0]])

theta0 = np.array([0.0, 0.0])
r0 = np.array([1.0, 1.0/3.0])
eps = 0.05
j, theta_trig, r_trig, theta_path, r_path = run_simplified_nuts(theta0, r0, eps)

distance_at_trigger = float(np.linalg.norm(theta_trig))

u = 0.98*np.exp(-hamiltonian(theta0, r0))
thetas_considered = theta_path[:2**j + 1]
rs_considered = r_path[:2**j + 1]
n_included = sum(np.exp(-hamiltonian(t, r)) >= u for t, r in zip(thetas_considered, rs_considered))

if __name__ == "__main__":  
    print(j)
    print(distance_at_trigger)
    print(n_included)
