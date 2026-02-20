import pandas as pd
import torch
import math
import matplotlib.pyplot as plt
import os

def student_logpdf_pytorch(x, nu):
    """Log pdf of Student-t distribution with nu degrees of freedom."""    
    return torch.lgamma(0.5*(nu+1)) - torch.lgamma(0.5*nu) - 0.5*math.log(math.pi) - 0.5*torch.log(nu) - 0.5*(nu+1)*torch.log(1 + x**2/nu)

def nll():
    resid = y - (alpha + beta * x)
    return -student_logpdf_pytorch(resid, nu).sum()

def closure():
    opt.zero_grad()
    loss = nll()
    loss.backward()
    return loss

fram = pd.read_csv(os.path.dirname(os.path.dirname(__file__))+'\\data\\raw\\fram.csv')
x = torch.tensor(fram['CHOL'].values, dtype=torch.double)
y = torch.tensor(fram['SBP'].values, dtype=torch.double)


x_mean, y_mean = x.mean(), y.mean()
beta_ols = ((x - x_mean)*(y - y_mean)).sum() / ((x - x_mean)**2).sum()
alpha_ols = y_mean - beta_ols * x_mean


nu = torch.tensor(5, dtype=torch.double)
alpha = torch.tensor(alpha_ols.item(), requires_grad=True)
beta  = torch.tensor(beta_ols.item(),  requires_grad=True)
opt = torch.optim.LBFGS([alpha, beta], lr=1, max_iter=500, line_search_fn="strong_wolfe")
opt.step(closure)
alpha_t, beta_t = alpha.detach().item(), beta.detach().item()
x_line = torch.linspace(x.min(), x.max(), 200)


if __name__ == "__main__":  
    plt.figure()
    plt.scatter(x.numpy(), y.numpy(), s=8)
    plt.xlabel("CHOL")
    plt.ylabel("SBP")

    print(f"OLS: alpha = {alpha_ols.item()}, beta = {beta_ols.item()}")
    print(f"Student-t (nu=5, sigma=1): alpha = {alpha_t}, beta = {beta_t}")

    plt.plot(x_line.numpy(), (alpha_ols + beta_ols*x_line).numpy(), label="OLS")
    plt.plot(x_line.numpy(), (alpha.detach() + beta.detach()*x_line).numpy(), label="Student-t ν=5")
    plt.legend()
    plt.show()
