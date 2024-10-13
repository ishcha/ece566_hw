import matplotlib.pyplot as plt
import numpy as np

# np.random.seed(1021)

alpha = 4
step_size_exp = [-1,-0.6]
x1 = 1
average_sgd = True # toggle to False for normal SGD

def get_derivative(x, alpha):
    if alpha == 4:
        return 4*(x**3)
    elif alpha == 2:
        return 2*x
    else: # alpha = 3/2
        return (x/abs(x))*3/2*(np.sqrt(abs(x)))

def step(x,k,step_size_e):
    deriv_sample = np.random.normal(get_derivative(x,alpha), 1)
    return x - ((k+1)**step_size_e)*deriv_sample

iters = 1000
k = np.arange(1,iters+1)
X_1 = [x1]
X_2 = [x1]

for i in range(1,iters):
    X_1.append(step(X_1[i-1],i,step_size_exp[0]))
    X_2.append(step(X_2[i-1],i,step_size_exp[1]))
    # print(X[i])
    
X_1 = [np.mean(X_1[:i]) for i in range(1,iters+1)] if average_sgd else X_1
X_2 = [np.mean(X_2[:i]) for i in range(1,iters+1)] if average_sgd else X_2
    
plt.plot(k, X_1, label=f'$k^{{{step_size_exp[0]}}}$')
plt.plot(k, X_2, label=f'$k^{{{step_size_exp[1]}}}$')
plt.plot(k, [0]*iters, 'r--')
plt.xlabel('Iteration ($k$)')
plt.ylabel('$x_k$')
plt.legend()
# plt.title(f'Convergence of SGD with $\\alpha = 4$ and step size $k^{step_size_exp}$')
plt.show()

plt.plot(np.log(k), np.abs(X_1), label=f'$k^{{{step_size_exp[0]}}}$')
plt.plot(np.log(k), np.abs(X_2), label=f'$k^{{{step_size_exp[1]}}}$')
plt.xlabel('Log of Iteration ($log k$)')
plt.ylabel('|$x_k$|')
plt.legend()
plt.show()