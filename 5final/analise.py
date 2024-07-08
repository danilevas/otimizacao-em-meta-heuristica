import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
from final import *

def roda_deepso(funcao, w_i, w_m, w_s, t_mut, t_com, com_hill_climb, plotar, printa_tudo):
    res_meu_deepso = list(meu_deepso(funcao, w_i, w_m, w_s, t_mut, t_com, com_hill_climb, plotar))
    print(f"Melhor posição: {pega_inteiro(res_meu_deepso[-1][0])}")
    print(f"Melhor valor: {res_meu_deepso[-1][1]}\n")
    if printa_tudo == True:
        for i in range(len(res_meu_deepso)):
            print(f"Iteração {i+1}: {pega_inteiro(res_meu_deepso[i][0])} = {res_meu_deepso[i][1]}")

def encontra_variaveis(funcao):
    valores = []
    variaveis = []
    for k in range(100):
        w_i = random.random()
        w_m = random.random() * (1 - w_i)
        w_s = random.random() * (1 - w_i - w_m)
        t_mut = random.random()
        t_com = random.random()

        valores.append(list(meu_deepso(rosenbrock_n, w_i=w_i, w_m=w_m, w_s=w_s, t_mut=t_mut, t_com=t_com, com_hill_climb=True))[-1][1])
        variaveis.append([w_i, w_m, w_s, t_mut, t_com])
        print(f"Iteração {k+1}: fitness {valores[-1]}")

    melhor_iter = np.argmin(valores)
    print(f"Melhor iteração: fitness {valores[np.argmin(valores)]}")
    print(f"\nw_i: {variaveis[melhor_iter][0]}\nw_m: {variaveis[melhor_iter][1]}\nw_s: {variaveis[melhor_iter][2]}\nt_mut: {variaveis[melhor_iter][3]}\nt_com: {variaveis[melhor_iter][4]}")

def analise(exs):
    # Análise dos algoritmos
    # Executamos todos 30 vezes com 100 gerações
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

    # Análise do HC
    plt.figure(figsize=(10, 6))
    plt.plot(list(range(20, 30)), log_hc_medias, linestyle='solid', label='Melhora')

    plt.title("Melhora de Xgb proporcionada pelo Hill Climbing no DEEPSO entre as gerações 20 e 30")
    plt.legend()
    plt.xlabel('Geração')
    plt.ylabel('Melhora do Hill Climbing')
    plt.yscale('log')

    plt.show()

    print(log_hc_medias)

roda_deepso(rosenbrock_n, w_i=0.53, w_m=0.2, w_s=0.17, t_mut=0.0027, t_com=0.959, com_hill_climb=True, plotar=True)