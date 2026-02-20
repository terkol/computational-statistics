# Regression

This folder contains small programs illustrating regression models with Bayesian and heavy-tailed likelihood variants. Topics include Bayesian linear regression with hierarchical priors sampled by Metropolis–Hastings, and robust regression under a Student-t noise model fit by numerical optimization.

## Programs

### `bayesian_regression.py`

Random-walk Metropolis–Hastings sampling for a Bayesian linear regression model with three predictors and unknown noise scale and coefficient scale.

Data are standardized before fitting.

Model:
$y \mid \beta,\sigma_x \sim \mathcal{N}(X\beta,\sigma_x^2 I_n)$

with priors
$\beta \mid \sigma_\beta \sim \mathcal{N}(0,\sigma_\beta^2 I_3),$
$\sigma_\beta \sim \mathrm{Gamma}(k,\theta),\quad \sigma_x \sim \mathrm{Gamma}(k,\theta)$
(using a shape–scale parameterization with constants omitted in the log posterior).

Sampling is performed on the transformed parameters
$s_\beta=\log\sigma_\beta, s_x=\log\sigma_x$
so that $\sigma_\beta=e^{s_\beta}$ and $\sigma_x=e^{s_x}$, with the appropriate Jacobian terms included in the log density.

The script runs multiple chains, computes split-$\hat{R}$ (Gelman–Rubin) diagnostics, and performs a second sampling round with a modified proposal scale for the parameter with the worst initial split-$\hat{R}$.

Dependencies: `numpy`, `pandas`.

Run: `python bayesian_regression.py`

### `student_t_regression.py`

Robust linear regression under a Student-t noise model with fixed degrees of freedom.

Model:
$y_i = \alpha + \beta x_i + \varepsilon_i,\quad \varepsilon_i \sim t_\nu(0,1)$
with $\nu$ fixed (default $\nu=5$) and unit scale.

The script computes the ordinary least squares (OLS) fit as a baseline, then estimates $(\alpha,\beta)$ under the Student-t likelihood by minimizing the negative log-likelihood using PyTorch autograd with an LBFGS optimizer, and plots both fitted lines.

Dependencies: `pandas`, `torch`, `matplotlib`.

Run: `python student_t_regression.py`