from numpy.random import randint
from numpy.random import rand
import numpy.random as npr
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Rosenbrock
def rosenbrock(X, a=1, b=100):
  x, y = X
  return (a - x)**2 + b * (y - x**2)**2

def plot(vetores):
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    x = np.linspace(-2.048, 2.048, 100)
    y = np.linspace(-2.048, 2.048, 100)
    X, Y = np.meshgrid(x, y)
    Z = rosenbrock([X,Y])
    ax.set_title('Rosenbrock 3D')
    ax.view_init(elev=50., azim=25)
    s = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='jet', edgecolor='none')
    fig.colorbar(s, shrink=0.75, aspect=15)

    # Plotando cada vetor como um ponto
    for v in vetores:
        x, y = v
        z = rosenbrock([x,y])
        ax.scatter(x, y, z, color='red')
    
    plt.show()

def plot_landscape(vetores):
    plt.figure(figsize=(10, 6))
    
    x = np.linspace(-2.048, 2.048, 100)
    y = np.linspace(-2.048, 2.048, 100)
    
    X, Y = np.meshgrid(x, y)
    Z = rosenbrock([X, Y])
    plt.contourf(X, Y, Z, levels=50, cmap='jet')

    # Plotando cada vetor como um ponto
    for v in vetores:
        plt.plot(v[0], v[1], 'ro')

    plt.xlabel('X')
    plt.ylabel('Y')
    
    plt.title('Rosenbrock Landscape')
    plt.colorbar(label='Z')

    plt.show()

vetores = [[-1.8, -1.5], [-1.6, 1], [0, 0], [1, 2], [0.55, 0.2]]
plot(vetores)
plot_landscape(vetores)