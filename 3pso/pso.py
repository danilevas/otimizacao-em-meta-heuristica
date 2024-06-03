import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define a função de Rastrigin
# def rastrigin(x):
#     n = len(x)
#     return 10*n + sum([xi**2 - 10*np.cos(2*np.pi*xi) for xi in x])

# Define a função de rosenbrock
def rosenbrock(X, a=1, b=100):
    x, y = X
    return (a - x)**2 + b * (y - x**2)**2

def sort_list(list1, list2):
	zipped_pairs = zip(list2, list1)
	z = [x for _, x in sorted(zipped_pairs)]
	return z

# Define o algoritmo de PSO
def pso(funcao, dim=2, limites=[-5.12, 5.12], num_particulas=30, max_iter=100, w=0.5, c1=1, c2=2):
    # Inicializa partículas e velocidades
    particulas = np.random.uniform(limites[0], limites[1], (num_particulas, dim))
    velocidades = np.zeros((num_particulas, dim))

    # Inicializa as melhores posições e os melhores valores de fitness
    melhores_posicoes = np.copy(particulas)
    melhor_fitness = np.array([funcao(p) for p in particulas])
    swarm_melhor_posicao = melhores_posicoes[np.argmin(melhor_fitness)]
    swarm_melhor_fitness = np.min(melhor_fitness)

    plota(funcao, limites, swarm_melhor_posicao, swarm_melhor_fitness)

    # Itera o número especificado de vezes, atualizando a velocidade e posição de cada partícula a cada iteração
    for i in range(max_iter):
        # Update velocidades
        r1 = np.random.uniform(0, 1, (num_particulas, dim))
        r2 = np.random.uniform(0, 1, (num_particulas, dim))
        velocidades = w * velocidades + c1 * r1 * (melhores_posicoes - particulas) + c2 * r2 * (swarm_melhor_posicao - particulas)

        # Atualiza as posições
        particulas += velocidades

        # Se algum índice sair dos limites, ele vai para o limite que ele estourou
        # e sua velocidade passa a ser 0
        for j in range(len(particulas)):
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
        
        if (i+1) % 25 == 0:
            plota(funcao, limites, swarm_melhor_posicao, swarm_melhor_fitness)

    # Retorna a melhor solução encontrada pelo algoritmo de PSO
    return swarm_melhor_posicao, swarm_melhor_fitness

def plota(funcao, limites, solucao, fitness):
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

def roda_30(funcao, dim=2, num_particulas=30, limites=[-5.12, 5.12], max_iter=100, w=0.5, c1=1, c2=2):
    # Executamos todos 30 vezes com 100 gerações
    bounds = [(-2.048, 2.048)]

    all_bests_meu_pso = []
    all_bests_pyswarm = []
    all_bests_pymoo = []

    for ex in range(30):
        bests_meu_pso = []

        # Roda todos os algoritmos
        res_meu_pso = list(de_meu_pso(fobj, bounds=bounds*2))
        print(f"Executou os meus algoritmos pela {ex+1}ª vez...")

        # Coloca o melhor de cada geração de cada algoritmo em uma lista
        for gen in range(len(res_meu_pso)):
            bests_meu_pso.append([res_meu_pso[gen][0], res_meu_pso[gen][1]])
            # print(f"{i}. [{result[i][0]}, {result[i][1]}] (taxa de melhora: {result[i][2]}%)")

        # Coloca o melhor de cada geração daquela execução na lista dos melhores de todas as execuções para cada algoritmo
        all_bests_meu_pso.append(bests_meu_pso)

    for ex in range(30):
        # Roda o algoritmo do Pymoo
        # Define a função como sendo a Rosenbrock
        funcao = get_problem("rosenbrock")
        meu_cb = MeuCallback()

        # Define o algoritmo genético
        algoritmo = DE(pop_size=100)
        res_pymoo = minimize(funcao, algoritmo, ('n_gen', 100), seed=1, verbose=False, callback=meu_cb.notify)
        print(f"Executou o algoritmo do Pymoo pela {ex+1}ª vez...")

        # Coloca o melhor de cada geração em uma lista
        all_bests_pymoo.append(meu_cb.best)

    meu_pso_melhores_inds_por_gen = []
    meu_pso_melhores_valores_por_gen = []
    pyswarm_melhores_inds_por_gen = []
    pyswarm_melhores_valores_por_gen = []
    pymoo_melhores_inds_por_gen = []
    pymoo_melhores_valores_por_gen = []

    i = 0
    for j in range(len(all_bests_meu_pso[i])):
        meu_pso_melhores_inds_por_gen.append([])
        meu_pso_melhores_valores_por_gen.append([])
        pyswarm_melhores_inds_por_gen.append([])
        pyswarm_melhores_valores_por_gen.append([])
        pymoo_melhores_inds_por_gen.append([])
        pymoo_melhores_valores_por_gen.append([])

        for i in range(len(all_bests_meu_pso)):
            meu_pso_melhores_inds_por_gen[j].append(all_bests_meu_pso[i][j][0])
            meu_pso_melhores_valores_por_gen[j].append(all_bests_meu_pso[i][j][1])
            pyswarm_melhores_inds_por_gen[j].append(all_bests_pyswarm[i][j][0])
            pyswarm_melhores_valores_por_gen[j].append(all_bests_pyswarm[i][j][1])
            pymoo_melhores_inds_por_gen[j].append(all_bests_pymoo[i][j][0])
            pymoo_melhores_valores_por_gen[j].append(all_bests_pymoo[i][j][1])

    meu_pso_medias = []
    pyswarm_medias = []
    pymoo_medias = []

    for i in range(len(meu_pso_melhores_valores_por_gen)):
        meu_pso_medias.append(sum(meu_pso_melhores_valores_por_gen[i])/len(meu_pso_melhores_valores_por_gen[i]))
        pyswarm_medias.append(sum(pyswarm_melhores_valores_por_gen[i])/len(pyswarm_melhores_valores_por_gen[i]))
        pymoo_medias.append(sum(pymoo_melhores_valores_por_gen[i])/len(pymoo_melhores_valores_por_gen[i]))

    meu_pso_melhores_inds_ult_gen_ordenados_por_fitness = sort_list(meu_pso_melhores_inds_por_gen[-1], meu_pso_melhores_valores_por_gen[-1])
    pyswarm_melhores_inds_ult_gen_ordenados_por_fitness = sort_list(pyswarm_melhores_inds_por_gen[-1], pyswarm_melhores_valores_por_gen[-1])
    pymoo_melhores_inds_ult_gen_ordenados_por_fitness = sort_list(pymoo_melhores_inds_por_gen[-1], pymoo_melhores_valores_por_gen[-1])
        
# Define as dimensões do problema
dim = 2

# Roda o algoritmo de PSO com a função de Rosenbrock
solucao, fitness = pso(rosenbrock, dim=dim)

# Printa a solução e valor de fitness
print('Solução:', solucao)
print('Fitness:', fitness)