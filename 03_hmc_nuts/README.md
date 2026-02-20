# HMC and NUTS

This folder contains small programs illustrating Hamiltonian Monte Carlo (HMC), leapfrog integration behavior, and a simplified No-U-Turn (NUTS) stopping criterion on a correlated Gaussian target. Examples include HMC sampling for Bayesian linear regression using PyTorch autograd, acceptance behavior versus step size for a Gaussian system, and a toy “doubling” trajectory with a U-turn trigger.

## Programs

### `hmc_sampling_linear_regression.py`

Hamiltonian Monte Carlo sampling for a Bayesian linear regression model using PyTorch for automatic differentiation.

Data model (standardized predictors $X \in \mathbb{R}^{n\times d}$ and response $Y\in\mathbb{R}^n$):
$Y \mid \beta,\sigma \sim \mathcal{N}(X\beta,\sigma^2 I_n).$

Parameters are sampled as
$\theta = (\beta, s_y)$
with $s_y = \log\sigma$ and $\sigma = e^{s_y}$.

Log-likelihood (up to an additive constant):
$\log p(Y\mid \beta,\sigma) = -n\log\sigma - \frac{1}{2\sigma^2}\lVert Y - X\beta\rVert_2^2.$

Priors:

Gaussian prior on regression coefficients:
$\beta \sim \mathcal{N}(0, 2^2 I_d).$

Gamma prior on $\sigma$ (shape–scale parameterization):
$\sigma \sim \mathrm{Gamma}(k,\theta)$
with log-density
$\log p(\sigma)= -\log\Gamma(k) - k\log\theta + (k-1)\log\sigma - \sigma/\theta.$
Sampling is performed in $s_y=\log\sigma$, so the Jacobian contributes an extra $+s_y$ term in the log density.

The script implements a basic HMC sampler with a fixed step size $\varepsilon$ and $L$ leapfrog steps, using Metropolis correction with Hamiltonian
$H(\theta,r) = U(\theta) + \tfrac12 r^\top r,$
where $U(\theta)=-\log p(\theta \mid X,Y)$.

Dependencies: `pandas`, `torch`.

Run: `python hmc_sampling_linear_regression.py`

### `leapfrog_hmc.py`

Leapfrog integration behavior and acceptance probability as a function of step size for a correlated 2D Gaussian target.

Target potential:
$U(\theta)=\tfrac12 \theta^\top \Sigma^{-1}\theta,$
with momentum $r\sim\mathcal{N}(0,I)$ and Hamiltonian
$H(\theta,r)=U(\theta)+\tfrac12 r^\top r.$

The script:

Sweeps a grid of step sizes $\varepsilon$ for fixed number of leapfrog steps $L$ and plots the implied Metropolis acceptance probability $\min(1,e^{-\Delta H})$.

Simulates a long leapfrog trajectory for a chosen $\varepsilon$ and visualizes the path in $(\theta_1,\theta_2)$ and the distance $\lVert \theta\rVert$ over steps (to highlight periodic motion in a quadratic potential).

Dependencies: `numpy`, `matplotlib`.

Run: `python leapfrog_hmc.py`

### `nuts.py`

Toy implementation of a simplified NUTS-style doubling procedure on a correlated 2D Gaussian target.

Target:
$\log \pi(\theta) = -\tfrac12 \theta^\top \Sigma^{-1}\theta,$
with gradient
$\nabla_\theta \log \pi(\theta) = -\Sigma^{-1}\theta.$

Starting from $(\theta_0,r_0)$, the code repeatedly doubles the number of leapfrog steps per segment:
$L = 2^{j-1}$
and checks a U-turn condition using endpoints of the new segment. It returns when the dot-product test indicates that the trajectory has started to turn back.

This is intended as an illustration of the U-turn idea and trajectory growth; it is not a full NUTS sampler (no proper slice-variable draw, no subtree recursion, no unbiased selection from the built trajectory).

Dependencies: `numpy`.

Run: `python nuts.py`