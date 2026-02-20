import numpy as np

np.random.seed(42)
k = 4
sigma = 2
target_halfwidth = 1e-2
se_target = target_halfwidth/1.96
variance_theta2 = (sigma**4)*(2*k)
N = int(np.ceil(variance_theta2/se_target**2))
z = np.random.normal(0, sigma, (N, k))
theta = np.linalg.norm(z, axis=1)
estimate = (theta**2).mean()
variance_sample = (theta**2).var(ddof=1)
standard_error = (variance_sample/N)**0.5
halfwidth = 1.96*standard_error
ci = (estimate-halfwidth, estimate+halfwidth)

v_value = estimate 
v_ci = (ci * np.ones(2))

if __name__ == '__main__':
    print("V: {}".format(v_value))
    print("95% confidence interval: {}".format(v_ci))