import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def gamma_logpdf_pytorch(x, alpha, beta):
    return (alpha*torch.log(beta) - torch.lgamma(alpha) + (alpha-1) * torch.log(x) - beta * x)

def nll_from_reparam(a, b, x):
    alpha = torch.exp(a)
    beta = torch.exp(b)
    return -gamma_logpdf_pytorch(x, alpha, beta).sum()

if __name__ == "__main__":
    torch.set_default_dtype(torch.double)
    df = pd.read_csv(os.path.dirname(os.path.dirname(__file__))+'\\data\\raw\\toydata.txt', header=None)

    
    data = torch.tensor(df.iloc[:,1].to_numpy(), dtype=torch.double).flatten()

    x_mean = data.mean()
    x_var = data.var(unbiased=True)
    alpha_init = (x_mean**2 / x_var).clamp(min=1e-6).item()
    beta_init = (alpha_init / x_mean).clamp(min=1e-6).item()
    a = torch.tensor(np.log(alpha_init), requires_grad=True)
    b = torch.tensor(np.log(beta_init), requires_grad=True)

    optimizer = torch.optim.LBFGS([a, b], lr=1, max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss = nll_from_reparam(a, b, data)
        loss.backward()
        return loss

    optimizer.step(closure)

    alpha_hat = torch.exp(a).item()
    beta_hat = torch.exp(b).item()
    print(f"alpha_hat (shape) = {alpha_hat}")
    print(f"beta_hat  (rate)  = {beta_hat}")

    with torch.no_grad():
        t_min = 0
        t_max = float(data.max().item() * 1.25)
        t_grid = torch.linspace(t_min, t_max, 400)
        pdf_grid = torch.exp(gamma_logpdf_pytorch(t_grid[1:], torch.exp(a), torch.exp(b)))

    plt.figure()
    plt.hist(data.numpy(), bins="fd", density=True)
    plt.plot(t_grid[1:].numpy(), pdf_grid.numpy())
    plt.show()