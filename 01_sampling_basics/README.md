# Sampling basics

This folder contains small programs illustrating core Monte Carlo sampling ideas used in computational statistics: direct Monte Carlo estimation, importance sampling, maximum likelihood via numerical optimization, Metropolis–Hastings (MH) Markov chain Monte Carlo, and a simple rejection-based approximate Bayesian computation (ABC) scheme.

## Programs
### `direct_sampling.py`

Direct Monte Carlo estimation of the second moment of a Gaussian norm. The script draws

$z \sim \mathcal{N}(0, \sigma^2 I_k)$,

forms $\theta = \lVert z \rVert_2$, and estimates

$V = \mathbb{E}[\theta^2] = \mathbb{E}!\left[\sum_{i=1}^k z_i^2\right] = k\sigma^2.$

It also computes an approximate 95% confidence interval using the sample variance of $\theta^2$ and the normal approximation:

$\widehat{V} \pm 1.96 \sqrt{\widehat{\mathrm{Var}}(\theta^2)/N}.$

Dependencies: `numpy`.

Run: `python direct_sampling.py`

### `importance_sampling.py`

Importance sampling for an expectation under a bimodal target distribution: a symmetric Gaussian mixture

$p(x) = \tfrac12 \mathcal{N}(-1, \mathrm{var}) + \tfrac12 \mathcal{N}(1, \mathrm{var}).$

The quantity of interest is

$\mathbb{E}_p[(X-1)^2].$

A Laplace proposal $q_b(x)$ (scale parameter $b$) is used, with log-weights

$\log w_i = \log p(x_i) - \log q_b(x_i), \quad x_i \sim q_b.$

The script picks $b$ from a small grid by maximizing a pilot effective sample size (ESS):

$\mathrm{ESS} = \dfrac{(\sum_i w_i)^2}{\sum_i w_i^2}.$

The self-normalized importance sampling estimator is

$\widehat{\mu} = \dfrac{\sum_i w_i f(x_i)}{\sum_i w_i}, \quad f(x)=(x-1)^2.$

Dependencies: `numpy`.

Run: `python importance_sampling.py`

### `max_likelihood_gamma.py`

Maximum likelihood estimation (MLE) for a Gamma distribution using PyTorch autograd and LBFGS optimization. With a Gamma(shape $\alpha$, rate $\beta$) model,

$\log p(x \mid \alpha,\beta) = \alpha \log \beta - \log \Gamma(\alpha) + (\alpha-1)\log x - \beta x,$

the script maximizes the log-likelihood over $(\alpha,\beta)$ using a reparameterization to enforce positivity:

$\alpha = e^a,;; \beta = e^b.$

It initializes $(\alpha,\beta)$ from method-of-moments and overlays the fitted density on a histogram.

Dependencies: `torch`, `pandas`, `numpy`, `matplotlib`.

Run: `python max_likelihood_gamma.py`

### `mh_1d_sampling.py`

Random-walk Metropolis–Hastings sampling in 1D for an unnormalized target density with log-density

$\log \tilde{\pi}(\theta) = 2\log|\cos \theta| - |\theta|^3.$

With proposal $\theta' = \theta + \varepsilon$, $\varepsilon \sim \mathcal{N}(0, s^2)$, the MH acceptance probability is

$\alpha(\theta,\theta') = \min{1, \exp(\log \tilde{\pi}(\theta') - \log \tilde{\pi}(\theta))}.$

The script tunes the proposal standard deviation $s$ by searching a grid for an acceptance rate near 0.5 and then estimates $\mathbb{E}[\cos\theta]$ from the post-burn-in samples.

Dependencies: `numpy`.

Run: `python mh_1d_sampling.py`

### `mh_2d_sampling.py`

Metropolis–Hastings sampling in 2D for two related targets.

A “ring” target centered at $\mu$ with preferred radius $r$:

$\log \tilde{\pi}(\theta) = -3(\lVert \theta-\mu\rVert_2 - r)^2.$

The same ring target with an additional $\ell_1$-type penalty:

$\log \tilde{\pi}(\theta) = -3(\lVert \theta-\mu\rVert_2 - r)^2 - |2\theta_1 - \theta_2|.$

The script estimates

$\mathbb{E}[\theta_1^2+\theta_2^2]$

and reports acceptance rates for hand-tuned proposal scales.

Dependencies: `numpy`.

Run: `python mh_2d_sampling.py`

### `rejection_abc.py`

Rejection-based approximate Bayesian computation (ABC) for an AR(1) time series model:

$X_{t+1} = a X_t + \varepsilon_t,\quad \varepsilon_t \sim \mathcal{N}(0,\sigma^2).$

Given observed data, the script computes a 2D summary statistic

$s(x) = \left(\frac{1}{n}\sum_{t=1}^n x_t^2,;; \frac{1}{n-1}\sum_{t=1}^{n-1} x_t x_{t+1}\right)$

and accepts parameter draws $(a,\sigma)$ from the prior if the simulated summaries are close in Euclidean distance:

$\lVert s(x_{\mathrm{sim}}) - s(x_{\mathrm{obs}})\rVert_2 \le \varepsilon.$

It returns accepted draws as an approximate posterior sample and reports posterior mean/SD for $(a,\sigma)$.

Dependencies: `numpy`, `pandas`.

Run: `python rejection_abc.py`