import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

# Define a função de rosenbrock
def rosenbrock_n(v, dim, a=1, b=100):
    if dim != len(v):
        raise Exception("Número de dimensões não bate com tamanho do vetor")
    valor = 0.
    for d in range(dim-1):
        valor += (a - v[d])**2 + b * (v[d+1] - v[d]**2)**2
    return valor

def rastrigin_n(v, dim, A=10):
    if dim != len(v):
        raise Exception("Número de dimensões não bate com tamanho do vetor")
    return A * dim + sum([(xi ** 2 - A * np.cos(2 * np.pi * xi)) for xi in v])

def pega_inteiro(array):
    my_str = "["
    for item in array:
        my_str += format(item, '.16f') + ", "
    my_str = my_str[:-2]
    my_str += "]"
    return my_str

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

def meu_normal():
    x = -1
    while x < 0:
        x = np.random.normal(0, 1)
    return x

def transpoe(arr):
    lista = []
    if isinstance(list(arr)[0], np.ndarray): # é vetor coluna
        for i in range(len(arr)):
            lista.append(arr[i][0])
    else:
        for i in range(len(arr)):
            lista.append([])
            lista[i].append(arr[i])
    return np.array(lista)

def plot(vetores, limites, dim, funcao):
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
    Z = funcao([X,Y], dim)
    ax.set_title('Rosenbrock 3D')
    ax.view_init(elev=50., azim=25)
    s = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='jet', edgecolor='none')
    fig.colorbar(s, shrink=0.75, aspect=15)

    # Plotando cada vetor como um ponto
    for v in vetores:
        x, y = v
        z = funcao([x,y], dim)
        ax.scatter(x, y, z, color='red')

    plt.show()

def plot_landscape(vetores, limites, dim, funcao):
    plt.figure(figsize=(9, 6))

    if abs(min(limites)) > abs(max(limites)):
        max_geral = abs(min(limites))
    else:
        max_geral = abs(max(limites))

    x = np.linspace(-max_geral, max_geral+2, 100)
    y = np.linspace(-max_geral, max_geral+2, 100)

    X, Y = np.meshgrid(x, y)
    Z = funcao([X, Y], dim)
    plt.contourf(X, Y, Z, levels=100, cmap='jet')

    # Plotando cada vetor como um ponto
    for v in vetores:
        plt.plot(v[0], v[1], 'ro')

    plt.xlabel('X')
    plt.ylabel('Y')

    plt.title('Rosenbrock Landscape')
    plt.colorbar(label='Z')

    plt.show()

def hill_climb(funcao, dim, posicao, fitness, fator_max):
    pos = np.copy(posicao)

    cand = pos + np.random.uniform(-fator_max, fator_max, dim)
    cand = np.clip(cand, -5, 5)
    fitness_cand = funcao(cand, dim)

    if fitness_cand < fitness:
        return cand, fitness_cand, True
    else:
        return posicao, fitness, False

# Define o algoritmo DEEPSO
def meu_deepso(funcao, w_i, w_m, w_s, t_mut, t_com, com_hill_climb, dim=2, limites=[-5, 5], num_particulas=30, max_iter=100, plotar=False):
    # Inicializa posições e velocidades das partículas
    posicoes = np.random.uniform(limites[0], limites[1], (num_particulas, dim))
    velocidades = np.zeros((num_particulas, dim))

    # Inicializa as melhores posições e os melhores valores de fitness
    valores_fitness = np.array([funcao(p, dim) for p in posicoes])
    x_bests = np.copy(posicoes)
    melhores_fitness = np.array([funcao(p, dim) for p in posicoes])
    x_gb = np.copy(x_bests[np.argmin(melhores_fitness)])
    x_gb_fitness = funcao(x_gb, dim)

    # Plota a população no plano 3D
    if plotar:
        limites_graf = min_max(posicoes)
        print("Geração 1")
        plot(posicoes, limites_graf, dim, funcao)
        plot_landscape(posicoes, limites_graf, dim, funcao)

    # Itera o número especificado de vezes, atualizando a velocidade e posição de cada partícula a cada iteração
    for iter in range(max_iter):
        for p in range(len(posicoes)):
            # Muta as variáveis
            w_i_mut = w_i + (t_mut * meu_normal())
            w_m_mut = w_m + (t_mut * meu_normal())
            w_s_mut = w_s + (t_mut * meu_normal())
            x_gb_mut = np.copy(x_gb) * (1 + (t_mut * meu_normal()))

            # Atualiza a matriz de comunicação
            C = np.zeros((dim, dim))
            for i in range(dim):
                for j in range(dim):
                    if i == j:
                        ri = meu_normal()
                        if ri < t_com:
                            C[i, j] = 1

            # Atualiza a velocidade
            parte1 = w_i_mut * velocidades[p]
            parte2 = w_m_mut * (x_bests[p] - posicoes[p])

            pre1_parte3 = (x_gb_mut - posicoes[p])
            # print(f"pre1_parte3: {pre1_parte3}")
            pre2_parte3 = transpoe(pre1_parte3)
            # print(f"pre2_parte3: {pre2_parte3}")

            # print(f"C: {C}")
            pre3_parte3 = C @ pre2_parte3
            # print(f"pre3_parte3: {pre3_parte3}")
            pre4_parte3 = transpoe(pre3_parte3)
            # print(f"pre4_parte3: {pre4_parte3}")

            parte3 = w_s_mut * pre4_parte3
            # print(f"parte1: {parte1}")
            # print(f"parte2: {parte2}")
            # print(f"parte3: {parte3}")
            # print()
            velocidades[p] = parte1 + parte2 + parte3

            # Atualiza a posição
            posicoes[p] += velocidades[p]

            # Se algum índice sair dos limites, ele vai para o limite que ele estourou e sua velocidade passa a ser 0
            for k in range(2):
                if posicoes[p][k] < limites[0]:
                    posicoes[p][k] = limites[0]
                    velocidades[p][k] = 0
                if posicoes[p][k] > limites[1]:
                    posicoes[p][k] = limites[1]
                    velocidades[p][k] = 0

            # Avalia o fitness da partícula
            valores_fitness[p] = funcao(posicoes[p], dim)

            # Atualiza o best e o global best
            if funcao(posicoes[p], dim) <= funcao(x_bests[p], dim):
                x_bests[p] = np.copy(posicoes[p])
                melhores_fitness[p] = funcao(posicoes[p], dim)
            if funcao(posicoes[p], dim) <= x_gb_fitness:
                # print(f"Xgb era {pega_inteiro(x_gb)} com fitness {x_gb_fitness} e agora é {pega_inteiro(posicoes[p])} com fitness {funcao(posicoes[p], dim)}")
                x_gb = np.copy(posicoes[p])
                x_gb_fitness = funcao(x_gb, dim)

        log_hc = ""
        if iter >= 20 and iter < 30 and com_hill_climb == True:
            fator_max = x_gb_fitness
            pos_hc, fitness_hc, funcionou = hill_climb(funcao, dim, x_gb, x_gb_fitness, fator_max)
            log_hc = x_gb_fitness - fitness_hc
            if funcionou == True:
                # print(f"HILLCLIMB! Xgb era {pega_inteiro(x_gb)} com fitness {x_gb_fitness} e agora é {pega_inteiro(pos_hc)} com fitness {fitness_hc}")
                x_gb = np.copy(pos_hc)
                x_gb_fitness = fitness_hc

        # Plota nas gerações 1, 25, 50 e 100
        if plotar and (iter == 24 or iter == 49 or iter == 99):
            limites_graf = min_max(posicoes)
            print(f"Geração {iter+1}")
            plot(posicoes, limites_graf, dim, funcao)
            plot_landscape(posicoes, limites_graf, dim, funcao)

        # Retorna a melhor solução encontrada pelo algoritmo naquela iteração
        yield x_gb, x_gb_fitness, log_hc

# Define o algoritmo DEEPSO
def meu_c_deepso(funcao, w_i, w_m, w_s, t_mut, t_com, com_hill_climb, dim=2, limites=[-5, 5], num_particulas=30, max_iter=100, plotar=False):
    # Inicializa posições e velocidades das partículas, e memória B
    posicoes = np.random.uniform(limites[0], limites[1], (num_particulas, dim))
    velocidades = np.zeros((num_particulas, dim))
    memoria_b = []

    # Inicializa as melhores posições e os melhores valores de fitness
    valores_fitness = np.array([funcao(p, dim) for p in posicoes])
    x_bests = np.copy(posicoes)
    melhores_fitness = np.array([funcao(p, dim) for p in posicoes])
    x_gb = np.copy(x_bests[np.argmin(melhores_fitness)])
    memoria_b.append(np.copy(x_gb))
    x_gb_fitness = funcao(x_gb, dim)

    # Plota a população no plano 3D
    if plotar:
        limites_graf = min_max(posicoes)
        print("Geração 1")
        plot(posicoes, limites_graf, dim, funcao)
        plot_landscape(posicoes, limites_graf, dim, funcao)

    # Itera o número especificado de vezes, atualizando a velocidade e posição de cada partícula a cada iteração
    for iter in range(max_iter):
        for p in range(len(posicoes)):
            # Muta as variáveis
            w_i_mut = w_i + (t_mut * meu_normal())
            w_m_mut = w_m + (t_mut * meu_normal())
            w_s_mut = w_s + (t_mut * meu_normal())
            x_gb_mut = np.copy(x_gb) * (1 + (t_mut * meu_normal()))

            # Atualiza a matriz de comunicação
            C = np.zeros((dim, dim))
            for i in range(dim):
                for j in range(dim):
                    if i == j:
                        ri = meu_normal()
                        if ri < t_com:
                            C[i, j] = 1
            # Cria o Xst
            # Xst = Xr + F(Xbest − Xr) + F(Xr1 − Xr2)
            # if len(memoria_b) >= 3:
            #     x_r = np.copy(random.choice(memoria_b))
            #     x_r1 = np.copy(random.choice(memoria_b))
            #     x_r2 = np.copy(random.choice(memoria_b))
            x_r = (np.copy(random.choice(posicoes)) + np.copy(random.choice(memoria_b))) / 2
            x_r1 = np.copy(random.choice(posicoes))
            x_r2 = np.copy(random.choice(posicoes))
            x_st = np.copy(x_r) + funcao(x_bests[p] - x_r, dim) + funcao(x_r1 - x_r2, dim)

            # Atualiza a velocidade
            parte1 = w_i_mut * velocidades[p]
            parte2 = w_m_mut * (x_st - posicoes[p])

            pre1_parte3 = (x_gb_mut - posicoes[p])
            pre2_parte3 = transpoe(pre1_parte3)

            pre3_parte3 = C @ pre2_parte3
            pre4_parte3 = transpoe(pre3_parte3)

            parte3 = w_s_mut * pre4_parte3
            velocidades[p] = parte1 + parte2 + parte3

            # Atualiza a posição
            posicoes[p] += velocidades[p]

            # Se algum índice sair dos limites, ele vai para o limite que ele estourou e sua velocidade passa a ser 0
            for k in range(2):
                if posicoes[p][k] < limites[0]:
                    posicoes[p][k] = limites[0]
                    velocidades[p][k] = 0
                if posicoes[p][k] > limites[1]:
                    posicoes[p][k] = limites[1]
                    velocidades[p][k] = 0

            # Avalia o fitness da partícula
            valores_fitness[p] = funcao(posicoes[p], dim)

            # Atualiza o best e o global best
            if funcao(posicoes[p], dim) <= funcao(x_bests[p], dim):
                x_bests[p] = np.copy(posicoes[p])
                melhores_fitness[p] = funcao(posicoes[p], dim)
            if funcao(posicoes[p], dim) <= x_gb_fitness:
                # print(f"Xgb era {pega_inteiro(x_gb)} com fitness {x_gb_fitness} e agora é {pega_inteiro(posicoes[p])} com fitness {funcao(posicoes[p], dim)}")
                x_gb = np.copy(posicoes[p])
                x_gb_fitness = funcao(x_gb, dim)

        log_hc = ""
        if iter >= 20 and iter < 30 and com_hill_climb == True:
            fator_max = x_gb_fitness
            pos_hc, fitness_hc, funcionou = hill_climb(funcao, dim, x_gb, x_gb_fitness, fator_max)
            log_hc = x_gb_fitness - fitness_hc
            if funcionou == True:
                # print(f"HILLCLIMB! Xgb era {pega_inteiro(x_gb)} com fitness {x_gb_fitness} e agora é {pega_inteiro(pos_hc)} com fitness {fitness_hc}")
                x_gb = np.copy(pos_hc)
                x_gb_fitness = fitness_hc

        # Plota nas gerações 1, 25, 50 e 100
        if plotar and (iter == 24 or iter == 49 or iter == 99):
            limites_graf = min_max(posicoes)
            print(f"Geração {iter+1}")
            plot(posicoes, limites_graf, dim, funcao)
            plot_landscape(posicoes, limites_graf, dim, funcao)

        # Coloca a melhor solução encontrada pelo algoritmo naquela iteração na memória B e retorna ela
        memoria_b.append(np.copy(x_gb))
        yield x_gb, x_gb_fitness, log_hc