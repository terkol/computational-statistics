# MCMC

This folder contains small programs illustrating Markov chain Monte Carlo (MCMC) methods and diagnostics. Topics include random-walk Metropolis–Hastings for Bayesian parameter inference, a simple Hamiltonian Monte Carlo (HMC) distance-scaling comparison, Metropolis sampling for a Student-t location model under different priors, and basic random-walk tuning/diagnostics for Bayesian logistic regression.

## Programs
### `mcmc_gamma.py`

Random-walk Metropolis–Hastings sampling for a Gamma(shape, rate) likelihood with Gamma priors on the positive parameters.

Model (data $x_i > 0$):
$ x_i \mid \alpha,\beta \sim \mathrm{Gamma}(\alpha,\beta) $
with density (shape–rate parameterization)
$\log p(x\mid \alpha,\beta)=\alpha\log\beta-\log\Gamma(\alpha)+(\alpha-1)\log x-\beta x.$

Sampling is performed on log-parameters

$a=\log\alpha$, 

$b=\log\beta$

so $\alpha=e^a$ and $\beta=e^b$.

A Gamma prior on $\alpha$ (and separately on $\beta$) induces a prior on $a$ (and $b$) that includes the Jacobian term:
$\log p(a) = \log p(e^a) + a.$

The script runs a Gaussian random-walk proposal in $(a,b)$, reports acceptance rate, and summarizes posterior means/SDs of $(\alpha,\beta)$.

Dependencies: `numpy`, `pandas`, `scipy`.

Run: `python mcmc_gamma.py`

### `mcmc_hmc_efficiency.py`

Distance-scaling comparison between random-walk Metropolis–Hastings (RW-MH) and a basic Hamiltonian Monte Carlo (HMC) integrator on a correlated 2D Gaussian target.

Target log-density:
$\log \pi(\theta) = -\tfrac12 \theta^\top \Sigma^{-1}\theta$
with strong correlation controlled by $\rho$.

RW-MH uses proposals

$\theta' = \theta + \sigma z$, 

$z\sim\mathcal{N}(0,I)$

and measures how the average distance from the start scales with step count.

HMC uses a standard leapfrog integrator with step size $\varepsilon$ and $L$ leapfrog steps, with Metropolis correction based on the Hamiltonian
$H(\theta,r)= -\log\pi(\theta) + \tfrac12 r^\top r,$
$r\sim\mathcal{N}(0,I).$

The script compares average distances for “short” vs “long” trajectories/runs to illustrate random-walk $\sqrt{n}$ scaling versus longer coherent moves.

Dependencies: `numpy`.

Run: `python mcmc_hmc_efficiency.py`

### `mcmc_student_t.py`

Metropolis–Hastings sampling for the location parameter of a Student-t observation model with fixed degrees of freedom.

Likelihood:
$y_i \mid \mu \sim t_{\nu}(\mu, 1)$
with $\nu$ fixed (default $\nu=5$).

Two priors are compared:

Normal prior: $\mu \sim \mathcal{N}(0,1)$, giving posterior log-density
$\log p(\mu\mid y) = \sum_i \log t_\nu(y_i;\mu,1) + \log \mathcal{N}(\mu;0,1).$

Uniform prior: $\mu \sim \mathrm{Unif}[-5,5]$, implemented by returning $-\infty$ outside the interval.

Dependencies: `numpy`, `pandas`, `scipy`.

Run: `python mcmc_student_t.py`

### `mcmc_tuning.py`

Random-walk Metropolis–Hastings for Bayesian logistic regression with a hierarchical prior, plus basic diagnostics (quantiles, split-$\hat{R}$, correlations, and pair plots).

Likelihood (binary outcomes $y_i\in{0,1}$):
$p(y_i=1 \mid \beta) = \mathrm{logit}^{-1}(\eta_i),\quad \eta_i=\beta_0 + x_i^\top \beta.$

The script uses a numerically stable form of the Bernoulli log-likelihood via the softplus function:
$\log \sigma(\eta) = -\mathrm{softplus}(-\eta),\quad \log(1-\sigma(\eta)) = -\mathrm{softplus}(\eta).$

Prior:

Student-t prior on regression coefficients (including intercept) with scale $\sigma_\beta$.

Gamma prior on the scale $\sigma_\beta$, sampled on $\log \sigma_\beta$ with the appropriate Jacobian.

During burn-in, the proposal scale is adapted in windows to target an acceptance rate around 0.3, using a multiplicative update
$s \leftarrow s\exp(c(\hat{a}-0.3)).$

After sampling, the script reports posterior quantiles, split-$\hat{R}$ across two chains, and visualizes pairwise scatter plots.

Dependencies: `numpy`, `pandas`, `scipy`, `matplotlib`.

Run: `python mcmc_tuning.py`