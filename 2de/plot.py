import numpy as np
import random
import matplotlib.pyplot as plt

def fobj(X, a=1, b=100):
  x, y = X
  return (a - x)**2 + b * (y - x**2)**2

def plot_adapta(vetores, limites):
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    x = np.linspace(limites[0], limites[1], 100)
    y = np.linspace(limites[2], limites[3], 100)
    X, Y = np.meshgrid(x, y)
    Z = fobj([X,Y])
    ax.set_title('Rosenbrock 3D')
    ax.view_init(elev=50., azim=25)
    s = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='jet', edgecolor='none')
    fig.colorbar(s, shrink=0.75, aspect=15)

    # Plotando cada vetor como um ponto
    for v in vetores:
        x, y = v
        z = fobj([x,y])
        ax.scatter(x, y, z, color='red')

    plt.show()

def plot_landscape_adapta(vetores, limites):
    plt.figure(figsize=(9, 6))

    x = np.linspace(limites[0], limites[1], 100)
    y = np.linspace(limites[2], limites[3], 100)

    X, Y = np.meshgrid(x, y)
    Z = fobj([X, Y])
    plt.contourf(X, Y, Z, levels=100, cmap='jet')

    # Plotando cada vetor como um ponto
    for v in vetores:
        plt.plot(v[0], v[1], 'ro')

    plt.xlabel('X')
    plt.ylabel('Y')

    plt.title('Rosenbrock Landscape')
    plt.colorbar(label='Z')

    plt.show()

def plot(vetores, limites):
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    if abs(min(limites)) > abs(max(limites)):
        max_geral = abs(min(limites))
    else:
        max_geral = abs(max(limites))
    
    x = np.linspace(-max_geral, max_geral, 100)
    y = np.linspace(-max_geral, max_geral, 100)
    X, Y = np.meshgrid(x, y)
    Z = fobj([X,Y])
    ax.set_title('Rosenbrock 3D')
    ax.view_init(elev=50., azim=25)
    s = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='jet', edgecolor='none')
    fig.colorbar(s, shrink=0.75, aspect=15)

    # Plotando cada vetor como um ponto
    for v in vetores:
        x, y = v
        z = fobj([x,y])
        ax.scatter(x, y, z, color='red')

    plt.show()

def plot_landscape(vetores, limites):
    plt.figure(figsize=(9, 6))

    if abs(min(limites)) > abs(max(limites)):
        max_geral = abs(min(limites))
    else:
        max_geral = abs(max(limites))
    
    x = np.linspace(-max_geral, max_geral, 100)
    y = np.linspace(-max_geral, max_geral, 100)

    X, Y = np.meshgrid(x, y)
    Z = fobj([X, Y])
    plt.contourf(X, Y, Z, levels=100, cmap='jet')

    # Plotando cada vetor como um ponto
    for v in vetores:
        plt.plot(v[0], v[1], 'ro')

    plt.xlabel('X')
    plt.ylabel('Y')

    plt.title('Rosenbrock Landscape')
    plt.colorbar(label='Z')

    plt.show()