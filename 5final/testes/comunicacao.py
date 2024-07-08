import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import time

def meu_normal():
    x = -1
    while x < 0:
        x = np.random.normal(0, 1)
    return x

dim = 50
t_com = 0.8

start = time.time()
C1 = np.eye(dim) * (np.random.rand(dim, dim) < t_com)
print(C1)
print(f"Tempo do método GPT: {time.time()-start}")

# Atualiza a matriz de comunicação
start = time.time()
C2 = np.zeros((dim, dim))
for i in range(dim):
    for j in range(dim):
        if i == j:
            ri = meu_normal()
            if ri < t_com:
                C2[i, j] = 1
print(C2)
print(f"Tempo do meu método: {time.time()-start}")