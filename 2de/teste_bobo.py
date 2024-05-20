import numpy as np

pop = [0, 3, 55, 12]
fitness = [0, 4, -1, 4]

def fobj(x):
    return x**2 - x

b = max(pop, key=fobj)
print(b)

best_idx = np.argmin(fitness)
print(best_idx)