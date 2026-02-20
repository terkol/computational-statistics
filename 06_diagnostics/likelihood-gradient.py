import torch


def sqexp_covariance(length_scale):
    dt = t_points[:, None] - t_points[None, :]
    return torch.exp(- (dt ** 2) / (2 * length_scale ** 2))

def logpdf_zero_mean(x, length_scale):
    K = sqexp_covariance(length_scale)
    _, logdet = torch.linalg.slogdet(K)
    alpha = torch.linalg.solve(K, x)
    quad = torch.dot(x, alpha)
    n = x.numel()
    return -0.5 * (n * torch.log(torch.tensor(2 * torch.pi)) + logdet + quad)

def K_of_ell(length_scale):
    return sqexp_covariance(length_scale)

t_points = torch.tensor([1.0, 2.0, 3.0])
x_vec = torch.tensor([1.0, 0.0, 1.0])
ell = 2.0
logpdf_val = logpdf_zero_mean(x_vec, torch.tensor(ell)).item()

h = 1e-3
fd_grad = (logpdf_zero_mean(x_vec, torch.tensor(ell + h)) - logpdf_zero_mean(x_vec, torch.tensor(ell))) / h
fd_grad_val = fd_grad.item()

ell_t = torch.tensor(ell, requires_grad=True)
logpdf_ad = logpdf_zero_mean(x_vec, ell_t)
logpdf_ad.backward()
exact_grad_val = ell_t.grad.item()

if __name__ == "__main__":

    print(logpdf_val)
    print(fd_grad_val)
    print(exact_grad_val)