import numpy as np 
import matplotlib.pyplot as plt
np.random.seed(42)

# load data 
data = np.loadtxt('data.txt')
print(data.shape)

def gaussian_pdf(x, mu, sigma):
    return 1 / np.sqrt(2 * np.pi * np.linalg.det(sigma)) * np.exp(-0.5 * np.dot(np.dot(x - mu, np.linalg.inv(sigma)), x - mu))

pis = []
mus = []
sigmas = []

# initialize parameters
pi = 0.5
mu = [np.random.rand(2), np.random.rand(2)]
sigma = [np.random.randn(2,2), np.random.randn(2,2)]

pis.append(pi)
mus.append(mu)
sigmas.append(sigma)

for i in range(100):
    # e-step 
    weights = np.zeros((data.shape[0], 2))
    for i in range(data.shape[0]):
        weights[i, 0] = gaussian_pdf(data[i], mu[0], sigma[0]) * pi
        weights[i, 1] = gaussian_pdf(data[i], mu[1], sigma[1]) * (1 - pi)
        weights[i] /= np.sum(weights[i])
        
    # m-step
    pi = np.sum(weights[:, 0]) / data.shape[0]
    mu = [np.dot(weights[:, 0], data) / np.sum(weights[:, 0]), np.dot(weights[:, 1], data) / np.sum(weights[:, 1])]
    sigma = [np.dot(weights[:, 0] * (data - mu[0]).T, data - mu[0]) / np.sum(weights[:, 0]), np.dot(weights[:, 1] * (data - mu[1]).T, data - mu[1]) / np.sum(weights[:, 1])]

    pis.append(pi)
    mus.append(mu)
    sigmas.append(sigma)
    
print(pis[-1])
print(mus[-1])
print(sigmas[-1])

error_pis = [np.abs(pis[i] - pis[-1]) for i in range(len(pis))]
error_mus0 = [np.linalg.norm(mus[i][0] - mus[-1][0]) for i in range(len(mus))]
error_mus1 = [np.linalg.norm(mus[i][1] - mus[-1][1]) for i in range(len(mus))]
error_sigmas0 = [np.linalg.norm(sigmas[i][0] - sigmas[-1][0]) for i in range(len(sigmas))]
error_sigmas1 = [np.linalg.norm(sigmas[i][1] - sigmas[-1][1]) for i in range(len(sigmas))]

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 10))

# First subplot
ax1.plot(error_pis)
ax1.set_title('$error_{\pi_1}$')

# Second subplot
ax2.plot(error_mus0, label='$\mu_1$')
ax2.plot(error_mus1, label='$\mu_2$')
ax2.set_title('$error_{\mu}$')
ax2.legend()

# Third subplot
ax3.plot(error_sigmas0, label='$\Sigma_1$')
ax3.plot(error_sigmas1, label='$\Sigma_2$')
ax3.set_title('$error_{\Sigma}$')
ax3.legend()

# Adjust spacing between subplots
plt.tight_layout()
plt.show()