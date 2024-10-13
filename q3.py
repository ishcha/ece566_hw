import numpy as np
import matplotlib.pyplot as plt

n = 1000

def get_exact(a):
    return 1/((100*a)**5)


def monte_carlo(a):
    vals = []
    for _ in range(n):
        x = np.random.uniform(0,1,10)
        l2_x = np.linalg.norm(x,2)
        vals.append(np.exp(-a*(l2_x**2)))
        
    return np.mean(vals)

all_a = np.linspace(0, 1, 100)

mc = [monte_carlo(a) for a in all_a]
exact = [get_exact(a) for a in all_a]
error = [abs(mc[i]-exact[i]) for i in range(len(mc))]

plt.plot(all_a, mc, label="Monte Carlo integral")
plt.plot(all_a, exact, label="Exact integral")
plt.plot(all_a, error, label="Error")
plt.legend()
plt.xlabel("a")
plt.show()