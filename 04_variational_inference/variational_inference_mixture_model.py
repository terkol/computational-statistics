import pandas as pd
import torch
import math
import os

def normal_logpdf_unit(x, mu):
    return -0.5*((x-mu)**2+log2pi)

def normal_logpdf(x, mu, sd):
    return -0.5*(((x-mu)/sd)**2+2*torch.log(sd)+log2pi)

def elbo_mc(mean_theta, x, logstd_theta, num_mc=8):
    std_theta = torch.exp(logstd_theta)
    eps = torch.randn(num_mc, 3, dtype=x.dtype)
    theta = mean_theta + std_theta * eps
    mu1, mu2, gamma = theta[:,0:1], theta[:,1:2], theta[:,2:3]
    w = torch.sigmoid(gamma)
    ll1 = torch.log(w)  + normal_logpdf_unit(x[None,:], mu1)
    ll2 = torch.log(1-w)+ normal_logpdf_unit(x[None,:], mu2)
    loglik = torch.logaddexp(ll1, ll2).sum(dim=1)
    logprior = (
        normal_logpdf(mu1.squeeze(1), 0, torch.sqrt(torch.tensor(10))).sum(dim=0) +
        normal_logpdf(mu2.squeeze(1), 0, torch.sqrt(torch.tensor(10))).sum(dim=0) +
        normal_logpdf(gamma.squeeze(1), 0, torch.tensor(1.78)).sum(dim=0))
    logq = normal_logpdf(theta, mean_theta, std_theta).sum(dim=1)
    elbo = (loglik + logprior - logq).mean()
    return elbo

torch.set_default_dtype(torch.double)
df = pd.read_csv(os.path.dirname(os.path.dirname(__file__))+'\\data\\raw\\mixture_data2.txt')
x = torch.tensor(df.values[:,0], dtype=torch.double)

log2pi = math.log(2*math.pi)

mean_theta = torch.zeros(3, requires_grad=True)
logstd_theta = torch.full((3,), -1.0, requires_grad=True)

if __name__ == "__main__": 
    opt = torch.optim.Adam([mean_theta, logstd_theta], lr=5e-3)
    for step in range(5000):
        opt.zero_grad(set_to_none=True)
        loss = -elbo_mc(mean_theta, x, logstd_theta, num_mc=16)
        loss.backward()
        opt.step()

    m_mu1, m_mu2, m_gamma = mean_theta.detach().tolist()
    s_mu1, s_mu2, s_gamma = torch.exp(logstd_theta.detach()).tolist()

    with torch.no_grad():
        S = 20000
        gamma_samples = torch.randn(S)*s_gamma + m_gamma
        w_samples = torch.sigmoid(gamma_samples)
        w_mean, w_std = w_samples.mean().item(), w_samples.std(unbiased=True).item()

    if m_mu1 > m_mu2:
        m_mu1, m_mu2 = m_mu2, m_mu1
        s_mu1, s_mu2 = s_mu2, s_mu1
        w_mean = 1 - w_mean

    print(f"q(mu1): mean = {m_mu1}, sd = {abs(s_mu1)}")
    print(f"q(mu2): mean = {m_mu2}, sd = {abs(s_mu2)}")
    print(f"w under q(gamma): mean = {w_mean}, sd = {abs(w_std)}")