import pyswarms as ps
import numpy as np

# Define a função de rosenbrock
def rosenbrock(X, a=1, b=100):
    x, y = X
    return (a - x)**2 + b * (y - x**2)**2

# Set-up hyperparameters
dimensions = 2
bounds = (np.array([-5.12, -5.12]), np.array([5.12, 5.12]))
options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
# Call instance of PSO
# optimizer = ps.single.GlobalBestPSO(n_particles=30, dimensions=2, options=options)
optimizer = ps.single.GlobalBestPSO(n_particles=30, dimensions=dimensions, options=options, bounds=bounds)
# Perform optimization
best_cost, best_pos = optimizer.optimize(rosenbrock, iters=100)
print(best_cost, best_pos)