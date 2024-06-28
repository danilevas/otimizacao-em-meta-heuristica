import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

# Define a função de rosenbrock_n
def rosenbrock_n(v, dim, a=1, b=100):
    if dim != len(v):
        raise Exception("Número de dimensões não bate com tamanho do vetor")
    valor = 0.
    for d in range(dim-1):
        valor += (a - v[d])**2 + b * (v[d+1] - v[d]**2)**2
    return valor

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

def plot(vetores, limites, funcao):
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
    Z = funcao([X,Y])
    ax.set_title('rosenbrock_n 3D')
    ax.view_init(elev=50., azim=25)
    s = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='jet', edgecolor='none')
    fig.colorbar(s, shrink=0.75, aspect=15)

    # Plotando cada vetor como um ponto
    for v in vetores:
        x, y = v
        z = funcao([x,y])
        ax.scatter(x, y, z, color='red')

    plt.show()

def plot_landscape(vetores, limites, funcao):
    plt.figure(figsize=(9, 6))

    if abs(min(limites)) > abs(max(limites)):
        max_geral = abs(min(limites))
    else:
        max_geral = abs(max(limites))

    x = np.linspace(-max_geral, max_geral+2, 100)
    y = np.linspace(-max_geral, max_geral+2, 100)

    X, Y = np.meshgrid(x, y)
    Z = funcao([X, Y])
    plt.contourf(X, Y, Z, levels=100, cmap='jet')

    # Plotando cada vetor como um ponto
    for v in vetores:
        plt.plot(v[0], v[1], 'ro')

    plt.xlabel('X')
    plt.ylabel('Y')

    plt.title('rosenbrock_n Landscape')
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
        plot(posicoes, limites_graf, funcao)
        plot_landscape(posicoes, limites_graf, funcao)

    # Itera o número especificado de vezes, atualizando a velocidade e posição de cada partícula a cada iteração
    for iter in range(max_iter):
        for p in range(len(posicoes)):
            # Muta as variáveis
            w_i_mut = w_i + (t_mut * np.random.normal(0, 1))
            w_m_mut = w_m + (t_mut * np.random.normal(0, 1))
            w_s_mut = w_s + (t_mut * np.random.normal(0, 1))
            x_gb_mut = np.copy(x_gb) * (1 + (t_mut * np.random.normal(0, 1)))

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

            pre1_parte3 = (x_gb_mut - posicoes[p])
            pre2_parte3 = np.array([[pre1_parte3[0]],[pre1_parte3[1]]])

            pre3_parte3 = C @ pre2_parte3
            pre4_parte3 = np.array([pre3_parte3[0][0], pre3_parte3[1][0]])

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
            plot(posicoes, limites_graf, funcao)
            plot_landscape(posicoes, limites_graf, funcao)

        # Retorna a melhor solução encontrada pelo algoritmo naquela iteração
        yield x_gb, x_gb_fitness, log_hc


res_meu_deepso = list(meu_deepso(rosenbrock_n, w_i=0.53, w_m=0.2, w_s=0.17, t_mut=0.0027, t_com=0.959, com_hill_climb=True, plotar=True))
print(f"Melhor posição: {pega_inteiro(res_meu_deepso[-1][0])}")
print(f"Melhor valor: {res_meu_deepso[-1][1]}")
# for i in range(len(res_meu_deepso)):
#     print(f"Iteração {i+1}: {pega_inteiro(res_meu_deepso[i][0])} = {res_meu_deepso[i][1]}")

# ANÁLISE
# Executamos todos 30 vezes com 100 gerações
exs = 30
bounds = [(-5, 5.)]

all_bests_deepso = []
all_bests_deepso_hc = []
all_logs_hc = []

for ex in range(exs):
    bests_deepso = []
    bests_deepso_hc = []

    # Roda o meu algoritmo
    res_deepso = list(meu_deepso(rosenbrock_n, w_i=0.53, w_m=0.2, w_s=0.17, t_mut=0.0027, t_com=0.959, com_hill_climb=False, plotar=False))
    res_deepso_hc = list(meu_deepso(rosenbrock_n, w_i=0.53, w_m=0.2, w_s=0.17, t_mut=0.0027, t_com=0.959, com_hill_climb=True, plotar=False))
    print(f"Executou os algoritmos pela {ex+1}ª vez...")

    # Coloca o melhor de cada geração do algoritmo em uma lista
    for gen in range(len(res_deepso)):
        bests_deepso.append([res_deepso[gen][0], res_deepso[gen][1]])
        bests_deepso_hc.append([res_deepso_hc[gen][0], res_deepso_hc[gen][1]])
        # print(f"{i}. [{result[i][0]}, {result[i][1]}] (taxa de melhora: {result[i][2]}%)")

    # Coloca o melhor de cada geração daquela execução na lista dos melhores de todas as execuções
    all_bests_deepso.append(bests_deepso)
    all_bests_deepso_hc.append(bests_deepso_hc)

    log_hc = []
    for i in range(len(res_deepso_hc)):
        if res_deepso_hc[i][2] != "":
            log_hc.append(res_deepso_hc[i][2])
    all_logs_hc.append(log_hc)

deepso_melhores_inds_por_gen = []
deepso_melhores_valores_por_gen = []
deepso_hc_melhores_inds_por_gen = []
deepso_hc_melhores_valores_por_gen = []
log_hc_melhora_por_gen = []

i = 0
for j in range(len(all_bests_deepso[i])):
    deepso_melhores_inds_por_gen.append([])
    deepso_melhores_valores_por_gen.append([])
    deepso_hc_melhores_inds_por_gen.append([])
    deepso_hc_melhores_valores_por_gen.append([])

    for i in range(len(all_bests_deepso)):
        deepso_melhores_inds_por_gen[j].append(all_bests_deepso[i][j][0])
        deepso_melhores_valores_por_gen[j].append(all_bests_deepso[i][j][1])
        deepso_hc_melhores_inds_por_gen[j].append(all_bests_deepso_hc[i][j][0])
        deepso_hc_melhores_valores_por_gen[j].append(all_bests_deepso_hc[i][j][1])

i = 0
for j in range(len(all_logs_hc[i])):
    log_hc_melhora_por_gen.append([])
    for i in range(len(all_logs_hc)):
        log_hc_melhora_por_gen[j].append(all_logs_hc[i][j])

deepso_medias = []
deepso_hc_medias = []
log_hc_medias = []

for i in range(len(deepso_melhores_valores_por_gen)):
    deepso_medias.append(sum(deepso_melhores_valores_por_gen[i])/len(deepso_melhores_valores_por_gen[i]))
    deepso_hc_medias.append(sum(deepso_hc_melhores_valores_por_gen[i])/len(deepso_hc_melhores_valores_por_gen[i]))

for i in range(len(log_hc_melhora_por_gen)):
    log_hc_medias.append(sum(log_hc_melhora_por_gen[i])/len(log_hc_melhora_por_gen[i]))

deepso_melhores_inds_ult_gen_ordenados_por_fitness = sort_list(deepso_melhores_inds_por_gen[-1], deepso_melhores_valores_por_gen[-1])
deepso_hc_melhores_inds_ult_gen_ordenados_por_fitness = sort_list(deepso_hc_melhores_inds_por_gen[-1], deepso_hc_melhores_valores_por_gen[-1])

plt.figure(figsize=(10, 6))
plt.plot(list(range(0, 100)), deepso_medias, linestyle='solid', label='DEEPSO')
plt.plot(list(range(0, 100)), deepso_hc_medias, linestyle='dashed', label='DEEPSO com Hill Climbing')

plt.legend()
plt.xlabel('Geração')
plt.ylabel('Média')
plt.yscale('log')

plt.show()

print()
print("Dados sobre os melhores indivíduos nas últimas gerações das 30 execuções do deepso:")
print(f"Média: {deepso_medias[-1]}")
print(f"Desvio Padrão: {np.std(deepso_melhores_valores_por_gen[-1])}")
print(f"Mediana: {np.median(deepso_melhores_valores_por_gen[-1])}")
print(f"Indivíduos que representam a Mediana: {deepso_melhores_inds_ult_gen_ordenados_por_fitness[14]}, {deepso_melhores_inds_ult_gen_ordenados_por_fitness[15]}")
print("Como temos 30 execuções (número par), o valor da mediana corresponde à média do fitness dos dois pontos acima")

print("\nDados sobre os melhores indivíduos nas últimas gerações das 30 execuções do deepso_hc:")
print(f"Média: {deepso_hc_medias[-1]}")
print(f"Desvio Padrão: {np.std(deepso_hc_melhores_valores_por_gen[-1])}")
print(f"Mediana: {np.median(deepso_hc_melhores_valores_por_gen[-1])}")
print(f"Indivíduos que representam a Mediana: {deepso_hc_melhores_inds_ult_gen_ordenados_por_fitness[14]}, {deepso_hc_melhores_inds_ult_gen_ordenados_por_fitness[15]}")
print("Como temos 30 execuções (número par), o valor da mediana corresponde à média do fitness dos dois pontos acima")

# ANÁLISE HILLCLIMB

plt.figure(figsize=(10, 6))
plt.plot(list(range(20, 30)), log_hc_medias, linestyle='solid', label='Melhora')

plt.title("Melhora de Xgb proporcionada pelo Hill Climbing no DEEPSO entre as gerações 20 e 30")
plt.legend()
plt.xlabel('Geração')
plt.ylabel('Melhora do Hill Climbing')
plt.yscale('log')

plt.show()

print(log_hc_medias)

# valores = []
# variaveis = []
# for k in range(100):
#     w_i = random.random()
#     w_m = random.random() * (1 - w_i)
#     w_s = random.random() * (1 - w_i - w_m)
#     t_mut = random.random()
#     t_com = random.random()

#     valores.append(list(meu_deepso(rosenbrock_n, w_i=w_i, w_m=w_m, w_s=w_s, t_mut=t_mut, t_com=t_com, com_hill_climb=True))[-1][1])
#     variaveis.append([w_i, w_m, w_s, t_mut, t_com])
#     print(f"Iteração {k+1}: fitness {valores[-1]}")

# melhor_iter = np.argmin(valores)
# print(f"Melhor iteração: fitness {valores[np.argmin(valores)]}")
# print(f"\nw_i: {variaveis[melhor_iter][0]}\nw_m: {variaveis[melhor_iter][1]}\nw_s: {variaveis[melhor_iter][2]}\nt_mut: {variaveis[melhor_iter][3]}\nt_com: {variaveis[melhor_iter][4]}")