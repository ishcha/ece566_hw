import numpy as np 
import matplotlib.pyplot as plt

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
mu = [np.zeros(2), np.ones(2)]
sigma = [np.eye(2), np.eye(2)]

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

plt.figure()
plt.plot(pis)
plt.title('pi')
plt.show()

plt.figure()
plt.plot([x[0] for x in mus])
plt.plot([x[1] for x in mus])
plt.title('mu')
plt.show()

plt.figure()
plt.plot([x[0] for x in sigmas])
plt.plot([x[1] for x in sigmas])
plt.title('sigma')
plt.show()