import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

limites=[-5, 5]
num_particulas = 30
dim = 2

posicoes = np.random.uniform(limites[0], limites[1], (num_particulas, dim))
print(random.choice(posicoes))

# # Define a função de rosenbrock
# def rosenbrock_n(v, dim, a=1, b=100):
#     if dim != len(v):
#         raise Exception("Número de dimensões não bate com tamanho do vetor")
#     valor = 0.
#     for d in range(dim-1):
#         valor += (a - v[d])**2 + b * (v[d+1] - v[d]**2)**2
#     return valor 

# print(rosenbrock_n([1.1,2,2,2,1,1,1], 7))

# def banana(x):
#     return x*3

# def chuchu(funcao, x):
#     if funcao == banana:
#         return "foi!"
#     else:
#         return "não foi"

# print(chuchu(banana, 2))