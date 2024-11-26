import pandas as pd
import numpy as np  
from scipy.special import digamma
from matplotlib import pyplot as plt
np.random.seed(0)

data = pd.read_csv('data.csv', sep=' ', header=None).to_numpy()
print(data.shape)

sample_mean = np.expand_dims(np.mean(data, axis=0), axis=0)

sample_covariance = np.cov(data.T)
print('empirical mean:', sample_mean, 'empirical covariance:', sample_covariance)

num_mixtures = 2
c_0 = 1*np.ones((num_mixtures,))
nu_0 = 2*np.ones((num_mixtures,))
mu_tilde = np.array([sample_mean+np.random.rand(2), sample_mean])
B_inverse_0 = np.array([nu_0*sample_covariance, nu_0*sample_covariance])
n = data.shape[0]
d = data.shape[1]
alpha_0 = 0.5*np.ones((num_mixtures,))
responsibilities = 0.5*np.ones((n,num_mixtures),dtype=float)


alphas = []
cs = []
nus = []
B_inverses = []
mu_bars = []

for _ in range(20):
    # evaluate parameters of 7.42
    alpha = alpha_0 + np.sum(responsibilities, axis=0)
    c = c_0 + np.sum(responsibilities,axis=0)
    nu = nu_0 + np.sum(responsibilities, axis=0)
    B_inverse = B_inverse_0 + np.array([(np.multiply(np.repeat(np.expand_dims(responsibilities[:,j],axis=1), d, axis=1), data).T)@data + c[j]*((mu_tilde[j].T)@mu_tilde[j]) for j in range(num_mixtures)])
    mu_bar = np.array([(c[j]*mu_tilde[j] + np.sum(np.multiply(np.repeat(np.expand_dims(responsibilities[:,j],axis=1), d, axis=1), data), axis=0,keepdims=True))/(c[j] + np.sum(responsibilities[:,j])) for j in range(num_mixtures)])
    alphas.append(alpha)
    cs.append(c)
    nus.append(nu)
    B_inverses.append(B_inverse)
    mu_bars.append(mu_bar)
    # evaluate expectations in 7.43
    e_omega = np.array([(digamma(alpha[ji]) - digamma(np.sum(alpha))) for ji in range(num_mixtures)])
    e_precision = np.array([np.sum([digamma((nu[j]+1-i)/2) for i in range(d)]) + d*np.log(2) - np.log(np.linalg.det(B_inverse[j])) for j in range(num_mixtures)])
    e_mu_yi = np.array([([d/c[j] + nu[j]*(data[i]-mu_bar[j])@(np.linalg.inv(B_inverse[j]))@((data[i]-mu_bar[j]).T) for j in range(num_mixtures)]) for i in range(n)]).squeeze()
    responsibilities = np.array([[e_omega[j] + 0.5*e_precision[j] - 0.5*e_mu_yi[i][j] for j in range(num_mixtures)] for i in range(n)])
    responsibilities = responsibilities/np.sum(responsibilities, axis=1,keepdims=True)
    
    
print("alpha:", alphas[-1])
print("c:", cs[-1])
print("nu:", nus[-1])
print("mu_bar:", mu_bars[-1])
print("B:", np.linalg.inv(B_inverses[-1]))
# exit()
error_alphas = [np.abs(alphas[i] - alphas[-1]) for i in range(len(alphas))]
error_cs = [np.abs(cs[i] - cs[-1]) for i in range(len(cs))]
error_nus = [np.abs(nus[i] - nus[-1]) for i in range(len(nus))]
error_mu_bars0 = [np.linalg.norm(mu_bars[i][0] - mu_bars[-1][0]) for i in range(len(mu_bars))]
error_mu_bars1 = [np.linalg.norm(mu_bars[i][1] - mu_bars[-1][1]) for i in range(len(mu_bars))]
error_B_inverses0 = [np.linalg.norm(B_inverses[i][0] - B_inverses[-1][0]) for i in range(len(B_inverses))]
error_B_inverses1 = [np.linalg.norm(B_inverses[i][1] - B_inverses[-1][1]) for i in range(len(B_inverses))]


fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1,5, figsize=(16, 4))

# First subplot
ax1.plot(error_alphas)
ax1.set_title('$error_{\\alpha}$')
ax2.plot(error_cs)
ax2.set_title('$error_{c}$')
ax3.plot(error_nus)
ax3.set_title('$error_{\\nu}$')

# Second subplot
ax4.plot(error_mu_bars0, label='$\\bar{\mu}_1$')
ax4.plot(error_mu_bars1, label='$\\bar{\mu}_2$')
ax4.set_title('$error_{\\bar{\mu}}$')
ax4.legend()

# Third subplot
ax5.plot(error_B_inverses0, label='$B^{-1}_1$')
ax5.plot(error_B_inverses1, label='$B^{-1}_2$')
ax5.set_title('$error_{B^{-1}}$')
ax5.legend()

# Adjust spacing between subplots
plt.tight_layout()
plt.show()