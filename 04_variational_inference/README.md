# Variational inference

This folder contains a small program illustrating variational inference (VI) for a simple Bayesian mixture model using the evidence lower bound (ELBO) and the reparameterization trick.

## Programs
### `variational_inference_mixture_model.py`

Mean-field variational inference for a two-component Gaussian location mixture with unknown component means and unknown mixing weight.

Model (unit-variance components):
$x_i \mid z_i,\mu_1,\mu_2 \sim \mathcal{N}(\mu_{z_i}, 1),\quad z_i \in {1,2},$
with mixing probability
$\Pr(z_i=1) = w.$

The mixing weight is parameterized via a logit variable $\gamma$:
$w = \sigma(\gamma)=\dfrac{1}{1+e^{-\gamma}}.$

Priors:
$\mu_1 \sim \mathcal{N}(0, 10),\quad \mu_2 \sim \mathcal{N}(0, 10),\quad \gamma \sim \mathcal{N}(0, 1.78^2).$

Variational family (mean-field Gaussian):
$q(\theta) = \mathcal{N}(\theta, m,\mathrm{diag}(s^2)),\quad \theta=(\mu_1,\mu_2,\gamma),$
parameterized by a mean vector $m$ and log-standard-deviation vector $\log s$.

The ELBO is estimated with Monte Carlo samples using the reparameterization trick:
$\theta = m + s \odot \varepsilon,\quad \varepsilon\sim\mathcal{N}(0,I),$
and optimized with Adam.

After optimization, the script reports variational means/SDs for $\mu_1$ and $\mu_2$, and estimates the mean/SD of $w=\sigma(\gamma)$ under $q(\gamma)$ by Monte Carlo sampling.

Dependencies: `pandas`, `torch`.

Run: `python variational_inference_mixture_model.py`