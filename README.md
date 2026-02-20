# Computational statistics

This repository contains a collection of small programs from coursework in computational statistics. The code is organized by topic and includes short demonstrations of Monte Carlo estimation, Markov chain Monte Carlo (MCMC), Hamiltonian Monte Carlo (HMC) and NUTS ideas, variational inference, Bayesian regression, and basic diagnostics.

Language: Python.

## Folder overview
### `01_sampling_basics`

Direct Monte Carlo, importance sampling, optimization-based maximum likelihood, Metropolis–Hastings sampling, and a simple rejection ABC example.

Programs include:

Direct sampling for estimating moments of a Gaussian norm with a normal-approximation confidence interval.

Importance sampling under a bimodal Gaussian mixture with proposal tuning via effective sample size.

Gamma MLE using PyTorch autograd and LBFGS.

1D and 2D Metropolis–Hastings examples (including proposal tuning via acceptance rate).

Rejection-based ABC for an AR(1) model using summary statistics and an $\varepsilon$ tolerance.

### `02_mcmc`

Random-walk Metropolis–Hastings for Bayesian inference, an HMC vs random-walk distance-scaling comparison, and basic tuning/diagnostics for a Bayesian logistic regression.

Programs include:

RW-MH sampling for Gamma(shape, rate) parameters in log space with Gamma priors.

A small experiment comparing average displacement under RW-MH vs naive HMC on a highly correlated Gaussian.

Metropolis sampling for the location parameter in a Student-t observation model under different priors.

RW-MH for logistic regression with hierarchical priors, burn-in adaptation of a global proposal scale, split-$\hat{R}$, posterior summaries, and correlation/pair plots.

### `03_hmc_nuts`

Hamiltonian Monte Carlo and NUTS-related building blocks, with both a practical HMC sampler and toy demonstrations of leapfrog behavior and a U-turn stopping trigger.

Programs include:

HMC sampling for Bayesian linear regression implemented in PyTorch using autograd for gradients (fixed step size and trajectory length).

Leapfrog integration for a correlated Gaussian target, showing acceptance probability versus step size and qualitative trajectory behavior.

A simplified “doubling” trajectory with a U-turn condition illustrating the NUTS stopping idea (not a full NUTS sampler).

### `04_variational_inference`

Mean-field variational inference for a Bayesian two-component Gaussian mixture model using a Monte Carlo ELBO estimator and the reparameterization trick.

Program includes:

Gaussian variational family over $(\mu_1,\mu_2,\gamma)$ with $w=\sigma(\gamma)$ as the mixture weight, optimized with Adam.

Post-processing to report variational means/SDs and the implied mean/SD of the mixing weight.

### `05_regression`

Regression examples including Bayesian linear regression via Metropolis–Hastings and a heavy-tailed Student-t regression fit by optimization.

Programs include:

Bayesian regression with unknown noise scale and coefficient scale (log-parameterized), multiple chains, and split-$\hat{R}$.

Student-t noise regression for $(\alpha,\beta)$ using PyTorch autograd and LBFGS (with fixed degrees of freedom and fixed unit scale).

### `06_diagnostics`

Basic diagnostics and gradient checking.

Programs include:

$\hat{R}$ and split-$\hat{R}$ for multi-chain output, plus an autocorrelation-based ESS estimator (implemented with a positive-sequence truncation heuristic).

A gradient check for a Gaussian-process log-likelihood with an RBF covariance: finite differences versus PyTorch autodiff.

Each topic folder contains a README.md describing the programs in that folder and how to run them.

### Running the code

Most scripts require `numpy`. Some additionally require `pandas`, `scipy`, `matplotlib`, or `torch` libraries. A minimal install that covers most folders is:

`pip install numpy pandas scipy matplotlib torch`

Run a script directly, for example:

`python 01_sampling_basics/mh_1d_sampling.py`

Data files

Several scripts load local data files stored under data/raw/ (e.g. `toydata.txt`, `mixture_data2.txt`, `babies2.txt`, `fram.csv`). Folder READMEs note which files are expected.