import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing as mp

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
T = 500

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


mc_iters = 100 # change to 1 for part a


def errors(my_iter, x):
    estimators = []
    predictors = []
    x = [x]*n

    for t in (range(1,T+1)):
        x_star = [sample_from_distribution(*transition_distribution(x[i], t)) for i in range(len(x))]
    
        pdf_obs = [stats.norm.pdf(observations[t-1],*observation_distribution(x_s)) for x_s in x_star]
    
        weights = [pdf_obs[i]/np.sum(pdf_obs) for i in range(len(pdf_obs))]
    
        idxs = range(len(x_star))
        idxs = np.random.choice(idxs, n, p=weights)
        x = [x_star[i] for i in idxs]
        estimators.append(np.mean(x))
        predictors.append(np.mean(x_star))
    
    estimation_error = [(estimators[i]-states[i])**2 for i in range(T)]
    predictor_error = [(predictors[i]-states[i])**2 for i in range(T)]
    print(my_iter)
    return estimation_error,predictor_error

num_processes = 10

with mp.Pool(processes=num_processes) as pool:
    results = pool.starmap_async(errors, [(k_itr, x) for k_itr in range(mc_iters)]).get()
    

mean_estimation_errors = [results[i][0] for i in range(mc_iters)]
mean_prediction_errors = [results[i][1] for i in range(mc_iters)]

mean_estimation_errors = [sum(elements) / len(elements) for elements in zip(*mean_estimation_errors)]
mean_prediction_errors = [sum(elements) / len(elements) for elements in zip(*mean_prediction_errors)]
    
# print(estimators[-1],predictors[-1], states[-1])
# exit()
fig, axs = plt.subplots(3, 1, figsize=(30, 15))

# Plot the states in the first subplot
axs[0].plot(range(1, 501), states, label='States', color='b')
axs[0].set_title('States')
axs[0].set_xlim(0, T)
# axs[0].legend()

# Plot the estimation error in the second subplot
axs[1].plot(range(1, 501), mean_estimation_errors, label='Estimation error', color='r')
axs[1].set_title('Estimation Error')
axs[1].set_xlim(0, T)
# axs[1].legend()

# Plot the predictor error in the third subplot
axs[2].plot(range(1, 501), mean_prediction_errors, label='Predictor error', color='g')
axs[2].set_title('Prediction Error')
# axs[2].legend()

# Adjust layout to prevent overlap
plt.tight_layout()
plt.xlim(0, T)
# plt.ylabel('Value')
# plt.xlabel('Time t')

# Show the plot
# plt.savefig('hw2/q5/b.png')
plt.show()