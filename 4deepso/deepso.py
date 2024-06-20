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
def meu_deepso(funcao, w_i, w_m, w_s, t_mut, t_com, dim=2, limites=[-5.12, 5.12], num_particulas=30, max_iter=100, c1=1, c2=2, plotar=False):
    # Inicializa posições e velocidades das partículas
    posicoes = np.random.uniform(limites[0], limites[1], (num_particulas, dim))
    velocidades = np.zeros((num_particulas, dim))

    # Inicializa as melhores posições e os melhores valores de fitness
    valores_fitness = np.array([funcao(p) for p in posicoes])
    x_bests = np.copy(posicoes)
    melhores_fitness = np.array([funcao(p) for p in posicoes])
    x_gb = x_bests[np.argmin(melhores_fitness)]

    # Plota a população no plano 3D
    if plotar:
        limites_graf = min_max(posicoes)
        print("Geração 1")
        plot(posicoes, limites_graf)
        plot_landscape(posicoes, limites_graf)

    # Itera o número especificado de vezes, atualizando a velocidade e posição de cada partícula a cada iteração
    for iter in range(max_iter):
        for p in range(len(posicoes)):
            # Muta as variáveis
            w_i_mut = w_i + (t_mut * np.random.normal(0, 1))
            w_m_mut = w_m + (t_mut * np.random.normal(0, 1))
            w_s_mut = w_s + (t_mut * np.random.normal(0, 1))
            x_gb_mut = x_gb * (1 + (t_mut * np.random.normal(0, 1)))

            # Atualiza a matriz de comunicação
            C = np.zeros((dim, dim))
            for i in range(dim):
                for j in range(dim):
                    if i == j:
                        ri = np.random.normal(0, 1)
                        if ri < t_com:
                            C[i, j] = 1

            # Atualiza a velocidade
            parte1 = w_i_mut * velocidades[p]
            parte2 = w_m_mut * (x_bests[p] - posicoes[p])
            parte3 = w_s_mut * C * (x_gb_mut - posicoes[p])
            velocidades[p] = parte1 + parte2 + parte3
            # Atualiza a posição
            posicoes[p] += velocidades[p]
            
            # Se algum índice sair dos limites, ele vai para o limite que ele estourou e sua velocidade passa a ser 0
            for k in range(2):
                if posicoes[j][k] < limites[0]:
                    posicoes[j][k] = limites[0]
                    velocidades[j][k] = 0
                if posicoes[j][k] > limites[1]:
                    posicoes[j][k] = limites[1]
                    velocidades[j][k] = 0
            
            # Avalia o fitness da partícula
            valores_fitness[p] = funcao(posicoes[p])

            # Atualiza o best e o global best
            if funcao(posicoes[p]) < funcao(x_bests[p]):
                x_bests[p] = posicoes[p]
                melhores_fitness[p] = funcao(posicoes[p])
            if funcao(posicoes[p]) < funcao(x_gb):
                x_gb = posicoes[p]

        # Plota nas gerações 1, 25, 50 e 100
        if plotar and (iter == 24 or iter == 49 or iter == 99):
            limites_graf = min_max(posicoes)
            print(f"Geração {iter+1}")
            plot(posicoes, limites_graf)
            plot_landscape(posicoes, limites_graf)

        # Retorna a melhor solução encontrada pelo algoritmo naquela iteração
        yield x_gb, funcao(x_gb)

res_meu_deepso = list(meu_deepso(himmelblau, w_i=0.3, w_m=0.3, w_s=0.4, t_mut=0.8, t_com=0.5))