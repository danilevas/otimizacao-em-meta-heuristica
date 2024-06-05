import pyswarms as ps
import numpy as np
import matplotlib.pyplot as plt

# Define a função de rosenbrock
def rosenbrock(X, a=1, b=100):
    x, y = X
    return (a - x)**2 + b * (y - x**2)**2

# Criando uma versão parametrizada da Rosenbrock
def rosenbrock_with_args(x, a=1, b=100, c=0):
    f = (a - x[:, 0]) ** 2 + b * (x[:, 1] - x[:, 0] ** 2) ** 2 + c
    return f

def sort_list(list1, list2):
	zipped_pairs = zip(list2, list1)
	z = [x for _, x in sorted(zipped_pairs)]
	return z

# Executamos todos 30 vezes com 100 gerações
exs = 30
bounds = [(-5.12, 5.12)]

all_bests_pyswarm = []

for ex in range(exs):
    # Set-up hyperparameters
    dimensions = 2
    bounds = (np.array([-5.12, -5.12]), np.array([5.12, 5.12]))
    options = {'c1': 1, 'c2': 2, 'w':0.5}

    # Call instance of PSO
    optimizer = ps.single.GlobalBestPSO(n_particles=30, dimensions=dimensions, options=options, bounds=bounds)
    
    # Perform optimization
    best_cost, best_pos = optimizer.optimize(rosenbrock_with_args, iters=100, a=1, b=100, c=0)

    # Access the history
    # cost_history = optimizer.cost_history  # List of costs at each iteration
    pos_history = optimizer.pos_history  # List of positions of particles at each iteration
    # best_pos_history = optimizer.swarm.history.best_pos

    bests_pyswarm = []
    for g in range(100):
        valores_iter = [rosenbrock(p) for p in pos_history[g]]
        ind_melhor = np.argmin(valores_iter)
        melhor_pos = pos_history[g][ind_melhor]
        if rosenbrock(melhor_pos) != min(valores_iter):
            raise Exception("Valores não batem")
        bests_pyswarm.append([list(melhor_pos), valores_iter[ind_melhor]])

    all_bests_pyswarm.append(bests_pyswarm)

pyswarm_melhores_inds_por_gen = []
pyswarm_melhores_valores_por_gen = []

i = 0
for j in range(len(all_bests_pyswarm[i])):
    pyswarm_melhores_inds_por_gen.append([])
    pyswarm_melhores_valores_por_gen.append([])

    for i in range(len(all_bests_pyswarm)):
        pyswarm_melhores_inds_por_gen[j].append(all_bests_pyswarm[i][j][0])
        pyswarm_melhores_valores_por_gen[j].append(all_bests_pyswarm[i][j][1])

pyswarm_medias = []

for i in range(len(pyswarm_melhores_valores_por_gen)):
    pyswarm_medias.append(sum(pyswarm_melhores_valores_por_gen[i])/len(pyswarm_melhores_valores_por_gen[i]))

pyswarm_melhores_inds_ult_gen_ordenados_por_fitness = sort_list(pyswarm_melhores_inds_por_gen[-1], pyswarm_melhores_valores_por_gen[-1])

plt.figure(figsize=(10, 6))
plt.plot(list(range(0, 100)), pyswarm_medias, linestyle='solid', label='Pyswarm')
plt.legend()
plt.xlabel('Geração')
plt.ylabel('Média')
plt.yscale('log')

plt.show()

print()
print("Dados sobre os melhores indivíduos nas últimas gerações das 30 execuções do Pyswarm:")
print(f"Média: {pyswarm_medias[-1]}")
print(f"Desvio Padrão: {np.std(pyswarm_melhores_valores_por_gen[-1])}")
print(f"Mediana: {np.median(pyswarm_melhores_valores_por_gen[-1])}")
print(f"Indivíduos que representam a Mediana: {pyswarm_melhores_inds_ult_gen_ordenados_por_fitness[14]}, {pyswarm_melhores_inds_ult_gen_ordenados_por_fitness[15]}")
print("Como temos 30 execuções (número par), o valor da mediana corresponde à média do fitness dos dois pontos acima")