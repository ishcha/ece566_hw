import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats

np.random.seed(0)

a = 0.5
b = 25
c = 8
omega = 1.2
U = (0,10) #TODO: check if this is variance or std dev
V = (0,1)

def transition_distribution(prev_x, t):
    mean = a*prev_x+b*(prev_x/(1+prev_x**2)) + c*np.cos(omega*(t-1)) + U[0]
    std_dev = np.sqrt(U[1])
    return (mean, std_dev)

def observation_distribution(prev_x):
    mean = (1/20)*prev_x**2 + V[0]
    std_dev = np.sqrt(V[1])
    return (mean, std_dev)

def sample_from_distribution(mean, std_dev):
    return np.random.normal(mean, std_dev)

n = 500
x = 0.1
T = 50

def true_observations(x0,T):
    x = x0
    observations = []
    states = []
    for t in range(1,T+1):
        x = sample_from_distribution(*transition_distribution(x, t))
        states.append(x)
        observations.append(sample_from_distribution(*observation_distribution(x)))
    return observations, states

observations, states = true_observations(x,T)
x = [x]*n

estimators = []
predictors = []

kde_t = [5, 50]

for t in tqdm(range(1,T+1)):
    
    x_star = [sample_from_distribution(*transition_distribution(x[i], t)) for i in range(len(x))]
    # print(states[t-1],observations[t-1])
    
    pdf_obs = [stats.norm.pdf(observations[t-1],*observation_distribution(x_s)) for x_s in x_star]
    
    weights = [pdf_obs[i]/np.sum(pdf_obs) for i in range(len(pdf_obs))]
    # print(sum(weights))
    # exit()
    
    idxs = range(len(x_star))
    idxs = np.random.choice(idxs, n, p=weights)
    x = [x_star[i] for i in idxs]
    estimators.append(np.mean(x))
    predictors.append(np.mean(x_star))
    
    if t in kde_t:
        kde = stats.gaussian_kde(x)
        x_range = np.linspace(min(x), max(x), 1000)

        kde_values = kde(x_range)

        plt.figure(figsize=(10, 6))
        plt.hist(x, bins=50, density=True, alpha=0.5, color='b')
        plt.plot(x_range, kde_values, 'r-', linewidth=2)
        # plt.title('PDF of particles at $t = {}$'.format(t))
        plt.xlabel('$X_t$')
        plt.ylabel('Density')
        plt.legend(['KDE', 'Histogram'])
        plt.grid(True)
        plt.show()