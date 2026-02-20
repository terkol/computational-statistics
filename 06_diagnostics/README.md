# Diagnostics

This folder contains small programs illustrating MCMC diagnostics and a basic gradient check for a Gaussian-process (GP) log-likelihood. Topics include Gelman–Rubin $\hat{R}$ and split-$\hat{R}$ convergence diagnostics, a simple effective sample size (ESS) calculation from autocorrelations, and comparison of finite-difference and automatic-differentiation gradients.

## Programs

### `convergence_diagnostics.py`

Computes convergence diagnostics for multiple MCMC chains stored as a matrix with shape (draws, chains).

Gelman–Rubin $\hat{R}$ compares between-chain and within-chain variability. For $m$ chains of length $n$, let $\bar{x}{\cdot j}$ be the mean of chain $j$, and $\bar{x}{\cdot\cdot}$ the mean of chain means. Define

$B = n,\mathrm{Var}(\bar{x}{\cdot j}), \qquad
W = \frac{1}{m}\sum{j=1}^m s_j^2,$

where $s_j^2$ is the within-chain variance. The marginal variance estimator is

$\widehat{\mathrm{Var}}^+ = \frac{n-1}{n}W + \frac{1}{n}B,$

and

$\hat{R} = \sqrt{\widehat{\mathrm{Var}}^+ / W}.$

Split-$\hat{R}$ applies the same calculation after splitting each chain into two halves, which can detect certain nonstationary behavior missed by standard $\hat{R}$.

The script also reports an ESS estimate for a single chain using an autocorrelation-based integrated autocorrelation time:

$\mathrm{ESS} \approx \frac{n}{\tau}, \qquad
\tau = 1 + 2\sum_{k\ge 1}\rho_k,$

with a “positive sequence” stopping rule that truncates the sum once adjacent autocorrelation pairs become non-positive.

Dependencies: `numpy`, `pandas`.

Run: `python convergence_diagnostics.py`

### `likelihood_gradient.py`

Gradient check for a zero-mean Gaussian log-likelihood with a squared-exponential (RBF) covariance function.

Given time points $t_1,\dots,t_n$, the covariance matrix is

$K_{ij}(\ell) = \exp!\left(-\frac{(t_i-t_j)^2}{2\ell^2}\right),$

and for $x\in\mathbb{R}^n$ the log density is

$\log p(x\mid \ell) =
-\frac12\left(n\log(2\pi) + \log|K(\ell)| + x^\top K(\ell)^{-1}x\right).$

The script computes:

the log density value at a chosen $\ell$,

a finite-difference approximation to $\frac{d}{d\ell}\log p(x\mid \ell)$,

the exact gradient from PyTorch automatic differentiation,

and prints all three for comparison.

Dependencies: `torch`.

Run: `python likelihood_gradient.py`