import pandas as pd
import torch
import os

torch.set_default_dtype(torch.float64)
rs = torch.Generator().manual_seed(42)

def standardize(x):
    if x.ndim == 1:
        mu = x.mean()
        sd = x.std(unbiased=True)
        return (x - mu) / (2 * sd)
    mu = x.mean(dim=0)
    sd = x.std(dim=0, unbiased=True)
    return (x - mu) / (2 * sd)

def gamma_logpdf(x, k, theta):
    return -torch.lgamma(k) - k*torch.log(theta) + (k-1)*torch.log(x) - x/theta

def log_joint(theta_vec, X, Y):
    beta = theta_vec[:d]
    s_y  = theta_vec[d]
    sigma = torch.exp(s_y)
    resid = Y - X.matmul(beta)
    ll = -Y.numel()*torch.log(sigma) - 0.5*resid.pow(2).sum()/(sigma*sigma)
    beta_prior = -0.5*(beta.pow(2)/(2**2)).sum()
    k = torch.tensor(10.0)
    theta_scale = torch.tensor(0.1)
    logp_sigma = gamma_logpdf(sigma, k, theta_scale)
    logp_sy = logp_sigma + s_y
    return ll + beta_prior + logp_sy

def potential_energy(theta_vec, X, Y):
    U = -log_joint(theta_vec, X, Y)
    return U

def leapfrog(theta, r, step_size, n_steps, X, Y):
    theta = theta.clone()
    r = r.clone()
    theta.requires_grad_(True)
    U = potential_energy(theta, X, Y)
    if not torch.isfinite(U):
        return theta.detach(), r.detach(), torch.tensor(float('inf'))
    grad = torch.autograd.grad(U, theta, create_graph=False)[0]
    if not torch.isfinite(grad).all():
        return theta.detach(), r.detach(), torch.tensor(float('inf'))

    r = r - 0.5*step_size*grad
    for i in range(n_steps):
        theta = (theta + step_size*r).detach().requires_grad_(True)
        U = potential_energy(theta, X, Y)
        if not torch.isfinite(U):
            return theta.detach(), r.detach(), torch.tensor(float('inf'))
        grad = torch.autograd.grad(U, theta, create_graph=False)[0]
        if i != n_steps - 1:
            r = r - step_size*grad
    r = r - 0.5*step_size*grad
    return theta.detach(), r.detach(), U.detach()

def hmc_sample(init_theta, X, Y, n_warmup=1000, n_samples=4000, L=40, step_size=0.01):
    dim = init_theta.numel()
    theta = init_theta.clone()
    samples = []
    accepts = 0
    total = 0
    step_size = torch.tensor(step_size, device=theta.device)
    for _ in range(n_warmup):
        r0 = torch.randn(dim, generator=rs, device=theta.device)
        U0 = potential_energy(theta, X, Y)
        K0 = 0.5 * r0.pow(2).sum()
        thetap, rp, Up = leapfrog(theta, r0, step_size, L, X, Y)
        if torch.isfinite(Up):
            Kp = 0.5 * rp.pow(2).sum()
            log_alpha = (U0 + K0) - (Up + Kp)
            accept_prob = torch.clamp(torch.exp(log_alpha), 0.0, 1.0)
            if torch.rand((), generator=rs, device=theta.device) < accept_prob:
                theta = thetap

    for _ in range(n_samples):
        r0 = torch.randn(dim, generator=rs, device=theta.device)
        U0 = potential_energy(theta, X, Y)
        K0 = 0.5 * r0.pow(2).sum()
        thetap, rp, Up = leapfrog(theta, r0, step_size, L, X, Y)
        total += 1
        if torch.isfinite(Up):
            Kp = 0.5 * rp.pow(2).sum()
            log_alpha = (U0 + K0) - (Up + Kp)
            accept_prob = torch.clamp(torch.exp(log_alpha), 0.0, 1.0)
            accept = torch.rand((), generator=rs, device=theta.device) < accept_prob
            accepts += int(accept.item())
            theta = thetap if accept else theta
        samples.append(theta.clone())

    accept_rate = accepts / max(1, total)
    return torch.stack(samples), step_size.item(), accept_rate

fram = pd.read_csv(os.path.dirname(os.path.dirname(__file__))+'\\data\\raw\\fram.csv')
print(fram)
x = torch.tensor(fram[['FRW','AGE','CHOL']].values.astype(float))
y = torch.tensor(fram['SBP'].values.astype(float))
xs = standardize(x)
ys = standardize(y)
n, d = xs.shape
Sigma_beta = 2.0**2

init = torch.zeros(d + 1)
init[-1] = torch.log(ys.std(unbiased=True))
samples, eps_used, acc = hmc_sample(init, xs, ys, n_warmup=1000, n_samples=4000, L=40, step_size=0.01)
beta_FRW = samples[:, 0]
s_y = samples[:, d]

if __name__ == "__main__":

    print("accept_rate:", acc, "step_size:", eps_used)
    print("beta_FRW mean/std:", beta_FRW.mean().item(), beta_FRW.std(unbiased=True).item())
    print("s_y mean/std:", s_y.mean().item(), s_y.std(unbiased=True).item())
