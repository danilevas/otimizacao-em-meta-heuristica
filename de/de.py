import numpy as np
import random
import matplotlib.pyplot as plt
import plot

def fobj(X, a=1, b=100):
    x, y = X
    return (a - x)**2 + b * (y - x**2)**2

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


def de_rand1(fobj, bounds, mut=0.8, crossp=0.7, popsize=20, gens=100, plotar=False):
    # Dimensões
    dimensions = len(bounds)
    # População inicial de vetores aleatórios (inicialmente com valores entre 0 e 1)
    pop_0_1 = np.random.rand(popsize, dimensions)

    # Normaliza os valores para entre os bounds
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    pop = min_b + pop_0_1 * diff

    # Calcula o fitness dos indivíduos
    fitness = np.asarray([fobj(ind) for ind in pop])
    # Pega o índice do indivíduo com menor fitness
    best_idx = np.argmin(fitness)
    # Pega o indivíduo com menor fitness e seu fitness
    best = pop[best_idx]
    best_fitness = fitness[best_idx]
    if plotar:
        limites_graf = min_max(pop)
        print("Geração 1")
        plot.plot(pop, limites_graf)
        plot.plot_landscape(pop, limites_graf)
    for gen in range(gens):
        # Contagem de quantas vezes o valor do novo indivíduo é melhor que o do antigo
        cont_melhorou = 0
        cont_igual = 0
        for j in range(popsize):
            # Pega os números de todos os índices menos o j
            idxs = [idx for idx in range(popsize) if idx != j]
            # Pega 3 vetores aleatórios que não j da população e executa a mutação
            # (mantendo os valores entre 0 e 1 com np.clip)
            a, b, c = pop_0_1[np.random.choice(idxs, 3, replace = False)]
            mutant = np.clip(a + mut * (b - c), 0, 1)
            # Cria um array com "dimensions" elementos e cada elemento tem chance "crossp" de ser True
            cross_points = np.random.rand(dimensions) < crossp
            # Se todos os elementos em cross_points forem False, bota um aleatório como verdadeiro
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dimensions)] = True

            # Nos índices em que o elemento de cross_points for True, ele pega a variável do mutante
            # Nos índices em que o elemento de cross_points for False, ele pega a variável do normal
            trial_0_1 = np.where(cross_points, mutant, pop_0_1[j])
            # Normaliza os valores para entre os bounds
            trial = min_b + trial_0_1 * diff

            # Pega o fitness do novo indivíduo
            f = fobj(trial)
            # Se a fitness do novo indivíduo for melhor que a do antigo, substitui o antigo pelo novo
            if f < fitness[j]:
                cont_melhorou += 1
                fitness[j] = f
                pop_0_1[j] = trial_0_1
                pop[j] = trial
                # Se a fitness do novo indivíduo for melhor que a melhor atual, coloca ela como a nova melhor
                if f < fitness[best_idx]:
                    best_idx = j
                    best = trial
                    best_fitness = f
            else:
                cont_igual += 1

        # Plota nas gerações 1, 25, 50 e 100
        if plotar and (gen == 24 or gen == 49 or gen == 99):
            limites_graf = min_max(pop)
            print(f"Geração {gen+1}")
            plot.plot(pop, limites_graf)
            plot.plot_landscape(pop, limites_graf)
        # Retorna o melhor indivíduo, seu fitness, e a porcentagem de melhora de indivíduos nessa geração
        pct_melhora = round((cont_melhorou/(cont_melhorou+cont_igual))*100, 2)
        yield best, fitness[best_idx], pct_melhora

def de_rand2(fobj, bounds, mut=0.8, crossp=0.7, popsize=20, gens=100):
    # Dimensões
    dimensions = len(bounds)
    # População inicial de vetores aleatórios (inicialmente com valores entre 0 e 1)
    pop_0_1 = np.random.rand(popsize, dimensions)

    # Normaliza os valores para entre -100 e 100
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    pop = min_b + pop_0_1 * diff
    
    # Calcula o fitness dos indivíduos
    fitness = np.asarray([fobj(ind) for ind in pop])
    # Pega o índice do indivíduo com menor fitness
    best_idx = np.argmin(fitness)
    # Pega o indivíduo com menor fitness
    best = pop[best_idx]
    for gen in range(gens):
        # Contagem de quantas vezes o valor do novo indivíduo é melhor que o do antigo
        cont_melhorou = 0
        cont_igual = 0
        for j in range(popsize):
            # Pega os números de todos os índices menos o j
            idxs = [idx for idx in range(popsize) if idx != j]
            # Pega 5 vetores aleatórios que não j da população e executa a mutação
            # (mantendo os valores entre 0 e 1 com np.clip)
            a, b, c, d, e = pop_0_1[np.random.choice(idxs, 5, replace = False)]
            mutant = np.clip(a + mut * (b - c) + mut * (d - e), 0, 1)
            # Cria um array com "dimensions" elementos e cada elemento tem chance "crossp" de ser True
            cross_points = np.random.rand(dimensions) < crossp
            # Se todos os elementos em cross_points forem False, bota um aleatório como verdadeiro
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dimensions)] = True

            # Nos índices em que o elemento de cross_points for True, ele pega a variável do mutante
            # Nos índices em que o elemento de cross_points for False, ele pega a variável do normal
            trial_0_1 = np.where(cross_points, mutant, pop_0_1[j])
            # Normaliza os valores para entre -100 e 100
            trial = min_b + trial_0_1 * diff
            
            # Pega o fitness do novo indivíduo
            f = fobj(trial)
            # Se a fitness do novo indivíduo for melhor que a do antigo, substitui o antigo pelo novo 
            if f < fitness[j]:
                cont_melhorou += 1
                fitness[j] = f
                pop_0_1[j] = trial_0_1
                # Se a fitness do novo indivíduo for melhor que a melhor atual, coloca ela como a nova melhor
                if f < fitness[best_idx]:
                    best_idx = j
                    best = trial
            else:
                cont_igual += 1
        
        # Retorna o melhor indivíduo, seu fitness quantas vezes os indivíduos novos foram melhores e piores que
        # os antigos nessa iteração
        pct_melhora = round((cont_melhorou/(cont_melhorou+cont_igual))*100, 2)
        yield best, fitness[best_idx], pct_melhora

def de_best1(fobj, bounds, mut=0.8, crossp=0.7, popsize=20, gens=100):
    # Dimensões
    dimensions = len(bounds)
    # População inicial de vetores aleatórios (inicialmente com valores entre 0 e 1)
    pop_0_1 = np.random.rand(popsize, dimensions)

    # Normaliza os valores para entre -100 e 100
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    pop = min_b + pop_0_1 * diff
    
    # Calcula o fitness dos indivíduos
    fitness = np.asarray([fobj(ind) for ind in pop])
    # Pega o índice do indivíduo com menor fitness
    best_idx = np.argmin(fitness)
    # Pega o indivíduo com menor fitness
    best = pop[best_idx]
    for gen in range(gens):
        # Contagem de quantas vezes o valor do novo indivíduo é melhor que o do antigo
        cont_melhorou = 0
        cont_igual = 0
        for j in range(popsize):
            # Pega os números de todos os índices menos o j
            idxs = [idx for idx in range(popsize) if idx != j]
            # Pega o indivíduo com menor fitness
            a = min(pop_0_1, key=fobj)
            # Pega 2 vetores aleatórios que não j da população e executa a mutação
            # (mantendo os valores entre 0 e 1 com np.clip)
            b, c = pop_0_1[np.random.choice(idxs, 2, replace = False)]
            mutant = np.clip(a + mut * (b - c), 0, 1)
            # Cria um array com "dimensions" elementos e cada elemento tem chance "crossp" de ser True
            cross_points = np.random.rand(dimensions) < crossp
            # Se todos os elementos em cross_points forem False, bota um aleatório como verdadeiro
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dimensions)] = True

            # Nos índices em que o elemento de cross_points for True, ele pega a variável do mutante
            # Nos índices em que o elemento de cross_points for False, ele pega a variável do normal
            trial_0_1 = np.where(cross_points, mutant, pop_0_1[j])
            # Normaliza os valores para entre -100 e 100
            trial = min_b + trial_0_1 * diff
            
            # Pega o fitness do novo indivíduo
            f = fobj(trial)
            # Se a fitness do novo indivíduo for melhor que a do antigo, substitui o antigo pelo novo 
            if f < fitness[j]:
                cont_melhorou += 1
                fitness[j] = f
                pop_0_1[j] = trial_0_1
                # Se a fitness do novo indivíduo for melhor que a melhor atual, coloca ela como a nova melhor
                if f < fitness[best_idx]:
                    best_idx = j
                    best = trial
            else:
                cont_igual += 1
        
        # Retorna o melhor indivíduo, seu fitness quantas vezes os indivíduos novos foram melhores e piores que
        # os antigos nessa iteração
        pct_melhora = round((cont_melhorou/(cont_melhorou+cont_igual))*100, 2)
        yield best, fitness[best_idx], pct_melhora

def de_best2(fobj, bounds, mut=0.8, crossp=0.7, popsize=20, gens=100):
    # Dimensões
    dimensions = len(bounds)
    # População inicial de vetores aleatórios (inicialmente com valores entre 0 e 1)
    pop_0_1 = np.random.rand(popsize, dimensions)

    # Normaliza os valores para entre -100 e 100
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    pop = min_b + pop_0_1 * diff
    
    # Calcula o fitness dos indivíduos
    fitness = np.asarray([fobj(ind) for ind in pop])
    # Pega o índice do indivíduo com menor fitness
    best_idx = np.argmin(fitness)
    # Pega o indivíduo com menor fitness
    best = pop[best_idx]
    for gen in range(gens):
        # Contagem de quantas vezes o valor do novo indivíduo é melhor que o do antigo
        cont_melhorou = 0
        cont_igual = 0
        for j in range(popsize):
            # Pega os números de todos os índices menos o j
            idxs = [idx for idx in range(popsize) if idx != j]
            # Pega o indivíduo com menor fitness
            a = min(pop_0_1, key=fobj)
            # Pega 4 vetores aleatórios que não j da população e executa a mutação
            # (mantendo os valores entre 0 e 1 com np.clip)
            b, c, d, e = pop_0_1[np.random.choice(idxs, 4, replace = False)]
            mutant = np.clip(a + mut * (b - c) + mut * (d - e), 0, 1)
            # Cria um array com "dimensions" elementos e cada elemento tem chance "crossp" de ser True
            cross_points = np.random.rand(dimensions) < crossp
            # Se todos os elementos em cross_points forem False, bota um aleatório como verdadeiro
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dimensions)] = True

            # Nos índices em que o elemento de cross_points for True, ele pega a variável do mutante
            # Nos índices em que o elemento de cross_points for False, ele pega a variável do normal
            trial_0_1 = np.where(cross_points, mutant, pop_0_1[j])
            # Normaliza os valores para entre -100 e 100
            trial = min_b + trial_0_1 * diff
            
            # Pega o fitness do novo indivíduo
            f = fobj(trial)
            # Se a fitness do novo indivíduo for melhor que a do antigo, substitui o antigo pelo novo 
            if f < fitness[j]:
                cont_melhorou += 1
                fitness[j] = f
                pop_0_1[j] = trial_0_1
                # Se a fitness do novo indivíduo for melhor que a melhor atual, coloca ela como a nova melhor
                if f < fitness[best_idx]:
                    best_idx = j
                    best = trial
            else:
                cont_igual += 1
        
        # Retorna o melhor indivíduo, seu fitness quantas vezes os indivíduos novos foram melhores e piores que
        # os antigos nessa iteração
        pct_melhora = round((cont_melhorou/(cont_melhorou+cont_igual))*100, 2)
        yield best, fitness[best_idx], pct_melhora

def de_currenttobest1(fobj, bounds, mut=0.8, crossp=0.7, popsize=20, gens=100):
    # Dimensões
    dimensions = len(bounds)
    # População inicial de vetores aleatórios (inicialmente com valores entre 0 e 1)
    pop_0_1 = np.random.rand(popsize, dimensions)

    # Normaliza os valores para entre -100 e 100
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    pop = min_b + pop_0_1 * diff
    
    # Calcula o fitness dos indivíduos
    fitness = np.asarray([fobj(ind) for ind in pop])
    # Pega o índice do indivíduo com menor fitness
    best_idx = np.argmin(fitness)
    # Pega o indivíduo com menor fitness
    best = pop[best_idx]
    for gen in range(gens):
        # Contagem de quantas vezes o valor do novo indivíduo é melhor que o do antigo
        cont_melhorou = 0
        cont_igual = 0
        for j in range(popsize):
            # Pega os números de todos os índices menos o j
            idxs = [idx for idx in range(popsize) if idx != j]
            # Pega o indivíduo com menor fitness
            b = min(pop_0_1, key=fobj)
            # Pega 3 vetores aleatórios que não j da população e executa a mutação
            # (mantendo os valores entre 0 e 1 com np.clip)
            a, c, d = pop_0_1[np.random.choice(idxs, 3, replace = False)]
            mutant = np.clip(a + mut * (b - a) + mut * (c - d), 0, 1)
            # Cria um array com "dimensions" elementos e cada elemento tem chance "crossp" de ser True
            cross_points = np.random.rand(dimensions) < crossp
            # Se todos os elementos em cross_points forem False, bota um aleatório como verdadeiro
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dimensions)] = True

            # Nos índices em que o elemento de cross_points for True, ele pega a variável do mutante
            # Nos índices em que o elemento de cross_points for False, ele pega a variável do normal
            trial_0_1 = np.where(cross_points, mutant, pop_0_1[j])
            # Normaliza os valores para entre -100 e 100
            trial = min_b + trial_0_1 * diff
            
            # Pega o fitness do novo indivíduo
            f = fobj(trial)
            # Se a fitness do novo indivíduo for melhor que a do antigo, substitui o antigo pelo novo 
            if f < fitness[j]:
                cont_melhorou += 1
                fitness[j] = f
                pop_0_1[j] = trial_0_1
                # Se a fitness do novo indivíduo for melhor que a melhor atual, coloca ela como a nova melhor
                if f < fitness[best_idx]:
                    best_idx = j
                    best = trial
            else:
                cont_igual += 1
        
        # Retorna o melhor indivíduo, seu fitness quantas vezes os indivíduos novos foram melhores e piores que
        # os antigos nessa iteração
        pct_melhora = round((cont_melhorou/(cont_melhorou+cont_igual))*100, 2)
        yield best, fitness[best_idx], pct_melhora

def compara_30():
    # Executamos todos 30 vezes com 100 gerações
    all_bests_rand1 = []
    all_bests_rand2 = []
    all_bests_best1 = []
    all_bests_best2 = []
    all_bests_currenttobest1 = []

    for ex in range(30):
        bests_rand1 = []
        bests_rand2 = []
        bests_best1 = []
        bests_best2 = []
        bests_currenttobest1 = []

        # Roda todos os algoritmos
        res_rand1 = list(de_rand1(fobj, bounds=[(-100, 100)] * 2))
        res_rand2 = list(de_rand2(fobj, bounds=[(-100, 100)] * 2))
        res_best1 = list(de_best1(fobj, bounds=[(-100, 100)] * 2))
        res_best2 = list(de_best2(fobj, bounds=[(-100, 100)] * 2))
        res_currenttobest1 = list(de_currenttobest1(fobj, bounds=[(-100, 100)] * 2))

        # Coloca o melhor de cada geração de cada algoritmo em uma lista
        for gen in range(len(res_rand1)):
            bests_rand1.append([res_rand1[gen][0], res_rand1[gen][1]])
            bests_rand2.append([res_rand2[gen][0], res_rand2[gen][1]])
            bests_best1.append([res_best1[gen][0], res_best1[gen][1]])
            bests_best2.append([res_best2[gen][0], res_best2[gen][1]])
            bests_currenttobest1.append([res_currenttobest1[gen][0], res_currenttobest1[gen][1]])
            # print(f"{i}. [{result[i][0]}, {result[i][1]}] (taxa de melhora: {result[i][2]}%)")
        
        # Coloca o melhor de cada geração daquela execução na lista dos melhores de todas as execuções para cada algoritmo
        all_bests_rand1.append(bests_rand1)
        all_bests_rand2.append(bests_rand2)
        all_bests_best1.append(bests_best1)
        all_bests_best2.append(bests_best2)
        all_bests_currenttobest1.append(bests_currenttobest1)

    rand1_melhores_inds_por_gen = []
    rand1_melhores_valores_por_gen = []
    rand2_melhores_inds_por_gen = []
    rand2_melhores_valores_por_gen = []
    best1_melhores_inds_por_gen = []
    best1_melhores_valores_por_gen = []
    best2_melhores_inds_por_gen = []
    best2_melhores_valores_por_gen = []
    currenttobest1_melhores_inds_por_gen = []
    currenttobest1_melhores_valores_por_gen = []

    i = 0
    for j in range(len(all_bests_rand1[i])):
        rand1_melhores_inds_por_gen.append([])
        rand1_melhores_valores_por_gen.append([])
        rand2_melhores_inds_por_gen.append([])
        rand2_melhores_valores_por_gen.append([])
        best1_melhores_inds_por_gen.append([])
        best1_melhores_valores_por_gen.append([])
        best2_melhores_inds_por_gen.append([])
        best2_melhores_valores_por_gen.append([])
        currenttobest1_melhores_inds_por_gen.append([])
        currenttobest1_melhores_valores_por_gen.append([])

        for i in range(len(all_bests_rand1)):
            rand1_melhores_inds_por_gen[j].append(all_bests_rand1[i][j][0])
            rand1_melhores_valores_por_gen[j].append(all_bests_rand1[i][j][1])
            rand2_melhores_inds_por_gen[j].append(all_bests_rand2[i][j][0])
            rand2_melhores_valores_por_gen[j].append(all_bests_rand2[i][j][1])
            best1_melhores_inds_por_gen[j].append(all_bests_best1[i][j][0])
            best1_melhores_valores_por_gen[j].append(all_bests_best1[i][j][1])
            best2_melhores_inds_por_gen[j].append(all_bests_best2[i][j][0])
            best2_melhores_valores_por_gen[j].append(all_bests_best2[i][j][1])
            currenttobest1_melhores_inds_por_gen[j].append(all_bests_currenttobest1[i][j][0])
            currenttobest1_melhores_valores_por_gen[j].append(all_bests_currenttobest1[i][j][1])

    rand1_medias = []
    rand2_medias = []
    best1_medias = []
    best2_medias = []
    currenttobest1_medias = []

    for i in range(len(rand1_melhores_valores_por_gen)):
        rand1_medias.append(sum(rand1_melhores_valores_por_gen[i])/len(rand1_melhores_valores_por_gen[i]))
        rand2_medias.append(sum(rand2_melhores_valores_por_gen[i])/len(rand2_melhores_valores_por_gen[i]))
        best1_medias.append(sum(best1_melhores_valores_por_gen[i])/len(best1_melhores_valores_por_gen[i]))
        best2_medias.append(sum(best2_melhores_valores_por_gen[i])/len(best2_melhores_valores_por_gen[i]))
        currenttobest1_medias.append(sum(currenttobest1_melhores_valores_por_gen[i])/len(currenttobest1_melhores_valores_por_gen[i]))

    rand1_melhores_inds_ult_gen_ordenados_por_fitness = sort_list(rand1_melhores_inds_por_gen[-1], rand1_melhores_valores_por_gen[-1])
    rand2_melhores_inds_ult_gen_ordenados_por_fitness = sort_list(rand2_melhores_inds_por_gen[-1], rand2_melhores_valores_por_gen[-1])
    best1_melhores_inds_ult_gen_ordenados_por_fitness = sort_list(best1_melhores_inds_por_gen[-1], best1_melhores_valores_por_gen[-1])
    best2_melhores_inds_ult_gen_ordenados_por_fitness = sort_list(best2_melhores_inds_por_gen[-1], best2_melhores_valores_por_gen[-1])
    currenttobest1_melhores_inds_ult_gen_ordenados_por_fitness = sort_list(currenttobest1_melhores_inds_por_gen[-1], currenttobest1_melhores_valores_por_gen[-1])

    # PLOT

    plt.plot(list(range(0, 100)), rand1_medias, label='rand1')
    plt.plot(list(range(0, 100)), rand2_medias, label='rand2')
    plt.plot(list(range(0, 100)), best1_medias, linestyle='dashed', label='best1')
    plt.plot(list(range(0, 100)), best2_medias, linestyle='dashed', label='best2')
    plt.plot(list(range(0, 100)), currenttobest1_medias, label='currenttobest1')
    plt.legend()
    plt.xlabel('Índice')
    plt.ylabel('Média')
    plt.show()

    print(len(rand1_melhores_valores_por_gen))
    print("Dados sobre os melhores indivíduos nas últimas gerações das 30 execuções do rand1:")
    print(f"Média: {rand1_medias[-1]}")
    print(f"Desvio Padrão: {np.std(rand1_melhores_valores_por_gen[-1])}")
    print(f"Mediana: {np.median(rand1_melhores_valores_por_gen[-1])}")
    print(f"Indivíduos que representam a Mediana: {rand1_melhores_inds_ult_gen_ordenados_por_fitness[14]}, {rand1_melhores_inds_ult_gen_ordenados_por_fitness[15]}")
    print("Como temos 30 execuções (número par), o valor da mediana corresponde à média do fitness dos dois pontos acima")

    print("\nDados sobre os melhores indivíduos nas últimas gerações das 30 execuções do rand2:")
    print(f"Média: {rand2_medias[-1]}")
    print(f"Desvio Padrão: {np.std(rand2_melhores_valores_por_gen[-1])}")
    print(f"Mediana: {np.median(rand2_melhores_valores_por_gen[-1])}")
    print(f"Indivíduos que representam a Mediana: {rand2_melhores_inds_ult_gen_ordenados_por_fitness[14]}, {rand2_melhores_inds_ult_gen_ordenados_por_fitness[15]}")
    print("Como temos 30 execuções (número par), o valor da mediana corresponde à média do fitness dos dois pontos acima")

    print("\nDados sobre os melhores indivíduos nas últimas gerações das 30 execuções do best1:")
    print(f"Média: {best1_medias[-1]}")
    print(f"Desvio Padrão: {np.std(best1_melhores_valores_por_gen[-1])}")
    print(f"Mediana: {np.median(best1_melhores_valores_por_gen[-1])}")
    print(f"Indivíduos que representam a Mediana: {best1_melhores_inds_ult_gen_ordenados_por_fitness[14]}, {best1_melhores_inds_ult_gen_ordenados_por_fitness[15]}")
    print("Como temos 30 execuções (número par), o valor da mediana corresponde à média do fitness dos dois pontos acima")

    print("\nDados sobre os melhores indivíduos nas últimas gerações das 30 execuções do best2:")
    print(f"Média: {best2_medias[-1]}")
    print(f"Desvio Padrão: {np.std(best2_melhores_valores_por_gen[-1])}")
    print(f"Mediana: {np.median(best2_melhores_valores_por_gen[-1])}")
    print(f"Indivíduos que representam a Mediana: {best2_melhores_inds_ult_gen_ordenados_por_fitness[14]}, {best2_melhores_inds_ult_gen_ordenados_por_fitness[15]}")
    print("Como temos 30 execuções (número par), o valor da mediana corresponde à média do fitness dos dois pontos acima")

    print("\nDados sobre os melhores indivíduos nas últimas gerações das 30 execuções do currenttobest1:")
    print(f"Média: {currenttobest1_medias[-1]}")
    print(f"Desvio Padrão: {np.std(currenttobest1_melhores_valores_por_gen[-1])}")
    print(f"Mediana: {np.median(currenttobest1_melhores_valores_por_gen[-1])}")
    print(f"Indivíduos que representam a Mediana: {currenttobest1_melhores_inds_ult_gen_ordenados_por_fitness[14]}, {currenttobest1_melhores_inds_ult_gen_ordenados_por_fitness[15]}")
    print("Como temos 30 execuções (número par), o valor da mediana corresponde à média do fitness dos dois pontos acima")

result = list(de_rand1(fobj, bounds=[(-100, 100)] * 2, plotar=True))
for i in range(len(result)):
    print(f"{i}. {result[i]}")