import matplotlib.pyplot as plt
import numpy as np

a = 0.125

def iteration(prev_iter):
    return prev_iter - 4*a*prev_iter**3

x0 = 1
iters = 1000

k = np.arange(1,iters+1)

x = np.zeros(iters)
x[0] = x0
for i in range(1, iters):
    x[i] = iteration(x[i-1])
    
x = [np.log(abs(x[i])) if x[i] > 0 else 0 for i in range(iters)]

plt.plot(np.log(k), x)
plt.scatter(np.log(k), x)
plt.xlabel('Log of Iteration ($log k$)')
plt.ylabel('log($x_k$)')
# plt.ylim(-.01, 0.1)
plt.title('Convergence of the sequence $x_k$')
plt.show()