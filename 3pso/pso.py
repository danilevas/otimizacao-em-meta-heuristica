import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the Rastrigin function
def rastrigin(x):
    n = len(x)
    return 10*n + sum([xi**2 - 10*np.cos(2*np.pi*xi) for xi in x])

# Define a função de rosenbrock
def rosenbrock(X, a=1, b=100):
    x, y = X
    return (a - x)**2 + b * (y - x**2)**2

# Define o algoritmo de PSO
def pso(funcao, dim=2, num_particulas=30, max_iter=100, w=0.5, c1=1, c2=2):
    # Inicializa partículas e velocidades
    particulas = np.random.uniform(-5.12, 5.12, (num_particulas, dim))
    velocidades = np.zeros((num_particulas, dim))

    # Inicializa as melhores posições e os melhores valores de fitness
    melhores_posicoes = np.copy(particulas)
    melhor_fitness = np.array([funcao(p) for p in particulas])
    swarm_melhor_posicao = melhores_posicoes[np.argmin(melhor_fitness)]
    swarm_melhor_fitness = np.min(melhor_fitness)

    # Itera o número especificado de vezes, atualizando a velocidade e posição de cada partícula a cada iteração
    for i in range(max_iter):
        # Update velocidades
        r1 = np.random.uniform(0, 1, (num_particulas, dim))
        r2 = np.random.uniform(0, 1, (num_particulas, dim))
        velocidades = w * velocidades + c1 * r1 * (melhores_posicoes - particulas) + c2 * r2 * (swarm_melhor_posicao - particulas)

        # Atualiza as posições
        particulas += velocidades

        # Avalia o fitness de cada partícula
        valores_fitness = np.array([funcao(p) for p in particulas])

        # Atualiza melhores posições e valores de fitness
        indices_melhorados = np.where(valores_fitness < melhor_fitness)
        melhores_posicoes[indices_melhorados] = particulas[indices_melhorados]
        melhor_fitness[indices_melhorados] = valores_fitness[indices_melhorados]
        if np.min(valores_fitness) < swarm_melhor_fitness:
            swarm_melhor_posicao = particulas[np.argmin(valores_fitness)]
            swarm_melhor_fitness = np.min(valores_fitness)

    # Retorna a melhor solução encontrada pelo algoritmo de PSO
    return swarm_melhor_posicao, swarm_melhor_fitness

# Define as dimensões do problema
dim = 2

# Roda o algoritmo de PSO com a função de Rosenbrock
solucao, fitness = pso(rosenbrock, dim=dim)

# Printa a solução e valor de fitness
print('Solução:', solucao)
print('Fitness:', fitness)

# Cria uma meshgrid para visualização
x = np.linspace(-5.12, 5.12, 100)
y = np.linspace(-5.12, 5.12, 100)
X, Y = np.meshgrid(x, y)
Z = rosenbrock([X, Y])

# Cria um plot 3D da função
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

# Plota a solução encontrada pelo algoritmo de PSO
ax.scatter(solucao[0], solucao[1], fitness, color='red')
plt.show()