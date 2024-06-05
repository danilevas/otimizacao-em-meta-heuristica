# Executamos todos 30 vezes com 100 gerações
exs = 30
bounds = [(-2.048, 2.048)]

all_bests_meu_pso = []
all_bests_pyswarm = []
all_bests_pymoo = []

for ex in range(exs):
    bests_meu_pso = []

    # Roda todos os algoritmos
    res_meu_pso = list(meu_pso(rosenbrock, bounds=bounds*2))
    print(f"Executou os meus algoritmos pela {ex+1}ª vez...")

    # Coloca o melhor de cada geração de cada algoritmo em uma lista
    for gen in range(len(res_meu_pso)):
        bests_meu_pso.append([res_meu_pso[gen][0], res_meu_pso[gen][1]])
        # print(f"{i}. [{result[i][0]}, {result[i][1]}] (taxa de melhora: {result[i][2]}%)")

    # Coloca o melhor de cada geração daquela execução na lista dos melhores de todas as execuções para cada algoritmo
    all_bests_meu_pso.append(bests_meu_pso)

for ex in range(exs):
    # Roda o algoritmo do Pymoo
    # Define a função como sendo a Rosenbrock
    funcao = get_problem("rosenbrock")
    meu_cb = MeuCallback()

    # Define o algoritmo genético
    algoritmo = PSO(pop_size=100)
    res_pymoo = minimize(funcao, algoritmo, ('n_gen', 100), seed=1, verbose=False, callback=meu_cb.notify)
    print(f"Executou o algoritmo do Pymoo pela {ex+1}ª vez...")

    # Coloca o melhor de cada geração em uma lista
    all_bests_pymoo.append(meu_cb.best)

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
    cost_history = optimizer.cost_history  # List of costs at each iteration
    pos_history = optimizer.pos_history  # List of positions of particles at each iteration

    bests_pyswarm = []
    for g in range(100):
        for p in pos_history[g]:
            if rosenbrock(p) == cost_history[g]:
                bests_pyswarm.append([p, cost_history[g]])
                break

    all_bests_pyswarm.append(bests_pyswarm)

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