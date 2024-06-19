import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define a função de rosenbrock
def rosenbrock(X, a=1, b=100):
    x, y = X
    return (a - x)**2 + b * (y - x**2)**2

def himmelblau(X):
    x, y = X
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

def sort_list(list1, list2):
	zipped_pairs = zip(list2, list1)
	z = [x for _, x in sorted(zipped_pairs)]
	return z

def min_max(pop):
    menor_x = menor_y = 10000
    maior_x = maior_y = -10000
    for i in range(len(pop)):
        if pop[i][0] < menor_x:
            menor_x = pop[i][0]
        if pop[i][0] > maior_x:
            maior_x = pop[i][0]
        if pop[i][1] < menor_y:
            menor_y = pop[i][1]
        if pop[i][1] > maior_y:
            maior_y = pop[i][1]

    return menor_x, maior_x, menor_y, maior_y

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

    x = np.linspace(-max_geral, max_geral+2, 100)
    y = np.linspace(-max_geral, max_geral+2, 100)
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

def plot_landscape(vetores, limites):
    plt.figure(figsize=(9, 6))

    if abs(min(limites)) > abs(max(limites)):
        max_geral = abs(min(limites))
    else:
        max_geral = abs(max(limites))

    x = np.linspace(-max_geral, max_geral+2, 100)
    y = np.linspace(-max_geral, max_geral+2, 100)

    X, Y = np.meshgrid(x, y)
    Z = rosenbrock([X, Y])
    plt.contourf(X, Y, Z, levels=100, cmap='jet')

    # Plotando cada vetor como um ponto
    for v in vetores:
        plt.plot(v[0], v[1], 'ro')

    plt.xlabel('X')
    plt.ylabel('Y')

    plt.title('Rosenbrock Landscape')
    plt.colorbar(label='Z')

    plt.show()

# Define o algoritmo DEEPSO
def meu_deepso(funcao, dim=2, limites=[-5.12, 5.12], num_particulas=30, max_iter=100, w=0.5, c1=1, c2=2, plotar=False):
    # Inicializa partículas e velocidades
    particulas = np.random.uniform(limites[0], limites[1], (num_particulas, dim))
    velocidades = np.zeros((num_particulas, dim))

    # Inicializa as melhores posições e os melhores valores de fitness
    melhores_posicoes = np.copy(particulas)
    melhor_fitness = np.array([funcao(p) for p in particulas])
    swarm_melhor_posicao = melhores_posicoes[np.argmin(melhor_fitness)]
    swarm_melhor_fitness = np.min(melhor_fitness)

    # Plota a população no plano 3D
    if plotar:
        limites_graf = min_max(particulas)
        print("Geração 1")
        plot(particulas, limites_graf)
        plot_landscape(particulas, limites_graf)

    # Itera o número especificado de vezes, atualizando a velocidade e posição de cada partícula a cada iteração
    for i in range(max_iter):
        # Atualiza as velocidades
        for p in range(len(particulas)):
            # Gerar candidato DEEPSO
            doador = np.copy(particula['posicao'])
            for j in range(dim):
                if np.random.rand() < CR:
                    doador[j] = particula['posicao'][j] + F * (particula['melhor_posicao'][j] - particula['posicao'][j])

            # Avaliar o candidato DEEPSO
            score_doador = himmelblau(doador)
            if score_doador < particula['melhor_score']:
                particula['melhor_score'] = score_doador
                particula['melhor_posicao'] = np.copy(doador)

            # Atualizar velocidade
            inercia = w * particula['velocidade']
            cognitivo = c1 * np.random.rand(dim) * (particula['melhor_posicao'] - particula['posicao'])
            social = c2 * np.random.rand(dim) * (melhor_global_posicao - particula['posicao'])
            particula['velocidade'] = inercia + cognitivo + social

            # Atualiza as posições
            particulas += velocidades

            # Se algum índice sair dos limites, ele vai para o limite que ele estourou e sua velocidade passa a ser 0
            for k in range(2):
                if particulas[j][k] < limites[0]:
                    particulas[j][k] = limites[0]
                    velocidades[j][k] = 0
                if particulas[j][k] > limites[1]:
                    particulas[j][k] = limites[1]
                    velocidades[j][k] = 0

        # Avalia o fitness de cada partícula
        valores_fitness = np.array([funcao(p) for p in particulas])

        # Atualiza melhores posições e valores de fitness
        indices_melhorados = np.where(valores_fitness < melhor_fitness)
        melhores_posicoes[indices_melhorados] = particulas[indices_melhorados]
        melhor_fitness[indices_melhorados] = valores_fitness[indices_melhorados]
        if np.min(valores_fitness) < swarm_melhor_fitness:
            swarm_melhor_posicao = particulas[np.argmin(valores_fitness)]
            swarm_melhor_fitness = np.min(valores_fitness)

        # Plota nas gerações 1, 25, 50 e 100
        if plotar and (i == 24 or i == 49 or i == 99):yy
            limites_graf = min_max(particulas)
            print(f"Geração {i+1}")
            plot(particulas, limites_graf)
            plot_landscape(particulas, limites_graf)

        # Retorna a melhor solução encontrada pelo algoritmo naquela iteração
        yield swarm_melhor_posicao, swarm_melhor_fitness