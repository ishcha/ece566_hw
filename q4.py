import numpy as np
# np.random.seed(42)

K = 1000
theta = 1

ns = [6, 100]


for n in ns:
    sample_n_y = np.random.normal(0, theta, n)
    estimation_errors = []
    for k in range(K):
        bootstrap_sample = np.random.choice(sample_n_y, n, replace=True)
        theta_hat = np.mean([bootstrap_sample[i]**2 for i in range(n)])
        estimation_errors.append(np.sqrt(n)*(theta_hat-theta))
        
    print(f"n = {n}:")
    print(f"Mean: {np.mean(estimation_errors)}")
    print(f"Variance: {np.var(estimation_errors)}")