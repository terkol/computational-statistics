import numpy as np
import pandas as pd
import os

def log_post(theta, X, y):
    s_beta, s_x, b1, b2, b3 = theta
    sig_b, sig_x = np.exp(s_beta), np.exp(s_x)
    beta = np.array([b1, b2, b3])
    mu = X @ beta
    n = y.size
    ll  = -0.5*np.sum((y-mu)**2)/(sig_x**2) - n*np.log(sig_x)
    lpb = -0.5*np.sum(beta**2)/(sig_b**2) - 3*np.log(sig_b)
    k, th = 2, 0.5
    lp_sbeta = (k-1)*np.log(sig_b) - sig_b/th + s_beta
    lp_sx    = (k-1)*np.log(sig_x) - sig_x/th + s_x
    return ll + lpb + lp_sbeta + lp_sx

def mh(theta, n_iter, prop_stds):
    th = theta.copy()
    d = th.size
    ps = np.full(d, prop_stds) if np.isscalar(prop_stds) else np.asarray(prop_stds)
    out = np.empty((n_iter, d))
    lp = log_post(th, x, y)
    for t in range(n_iter):
        prop = th + np.random.normal(0, ps, d)
        lp_prop = log_post(prop, x, y)
        if np.log(np.random.rand()) < lp_prop - lp:
            th, lp = prop, lp_prop
        out[t] = th
    return out

def split_rhat(chains):
    C, N, D = chains.shape
    halves = np.concatenate(np.split(chains, 2, axis=1), axis=0)
    m = halves.mean(1)
    W = halves.var(1, ddof=1).mean(0)
    B = halves.shape[1] * m.var(0, ddof=1)
    var_hat = ((halves.shape[1]-1)/halves.shape[1])*W + (1/halves.shape[1])*B
    return np.sqrt(var_hat / W)

babies_full = pd.read_csv(os.path.dirname(os.path.dirname(__file__))+'\\data\\raw\\babies2.txt', sep='\t')
babies = babies_full.loc[(babies_full['age']>=30).values,:]
x = babies[['age', 'gestation', 'weight']].values.astype(float)
y = babies[['bwt']].values.astype(float).squeeze()
x -= np.mean(x, 0)
x /= np.std(x, 0)
y -= np.mean(y, 0)
y /= np.std(y, 0)
inits = np.array([[ 1.67272789, -0.02134183, -0.78116796, -1.77420787],
                [-0.6030697,   0.4464263,   0.3627991,  -0.7342916 ],
                [-1.32685026, -0.37427079, -0.06744599,  0.21175491],
                [ 0.87702047, -0.54171875, -0.44736686, -2.39045985],
                [ 1.00631791, -1.34638203, -0.40343913, -0.64535026]])
chains1 = []
for j in range(inits.shape[1]):
    th0 = inits[:, j]
    sam = mh(th0, n_iter=5000, prop_stds=0.04)[2500:]
    chains1.append(sam)
chains1 = np.stack(chains1)
rhat1 = split_rhat(chains1)

worst_idx = int(np.argmax(rhat1))

prop2 = np.full(5, 0.04); prop2[worst_idx] = 0.4
chains2 = []
for j in range(inits.shape[1]):
    th0 = inits[:, j]
    sam = mh(th0, n_iter=5000, prop_stds=prop2)[2500:]
    chains2.append(sam)
chains2 = np.stack(chains2)
rhat2 = split_rhat(chains2)

post_mean_sbeta = chains2.reshape(-1, 5)[:, 0].mean()

if __name__ == "__main__":

    print("split-Rhat (0.04 each):", rhat1)
    print("max split-Rhat:", rhat1[worst_idx], "at index", worst_idx)

    print("split-Rhat (0.4 on idx", worst_idx, "):", rhat2)
    print("max split-Rhat (round 2):", rhat2.max())

    print("Posterior mean s_beta (round 2):", post_mean_sbeta)