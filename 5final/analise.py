import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
from final import *

def roda(algoritmo, funcao, w_i, w_m, w_s, t_mut, t_com, dim, limites, num_particulas, max_iter, com_hill_climb, plotar, printa_tudo):
    res_meu_deepso = list(algoritmo(funcao, w_i, w_m, w_s, t_mut, t_com, com_hill_climb, dim, limites, num_particulas, max_iter, plotar))
    print(f"Melhor posição: {pega_inteiro(res_meu_deepso[-1][0])}")
    print(f"Melhor valor: {res_meu_deepso[-1][1]}\n")
    if printa_tudo == True:
        for i in range(len(res_meu_deepso)):
            print(f"Iteração {i+1}: {pega_inteiro(res_meu_deepso[i][0])} = {res_meu_deepso[i][1]}")

def encontra_variaveis(algoritmo, funcao, dim, conjuntos, sufixo):
    exs = 3
    valores = []
    variaveis = []
    for k in range(conjuntos):
        w_i = random.random()
        w_m = random.random() * (1 - w_i)
        w_s = 1 - w_i - w_m
        t_mut = random.random()
        t_com = random.random()

        variaveis.append([w_i, w_m, w_s, t_mut, t_com])
        testes = []
        for ex in range(exs):
            testes.append(list(algoritmo(funcao, w_i=w_i, w_m=w_m, w_s=w_s, t_mut=t_mut, t_com=t_com, dim=dim))[-1][1])
        media = sum(testes)/len(testes)
        valores.append(media)
        print(f"Conjunto {k+1}: fitness médio {media}")

    melhor_iter = np.argmin(valores)
    print(f"\nMelhor conjunto para {sufixo}: fitness {valores[np.argmin(valores)]}")
    print(f"w_i={variaveis[melhor_iter][0]}, w_m={variaveis[melhor_iter][1]}, w_s={variaveis[melhor_iter][2]}, t_mut={variaveis[melhor_iter][3]}, t_com={variaveis[melhor_iter][4]}\n")

def analise(exs, algoritmo, funcao, w_i, w_m, w_s, t_mut, t_com, dim):
    # Análise dos algoritmos
    # Executamos todos 30 vezes com 100 gerações
    bounds_rosenbrock = [(-5., 5.)]
    bounds_rastrigin = [(-5.12, 5.12)]

    all_bests = []
    all_logs_hc = []

    for ex in range(exs):
        bests = []

        # Roda o algoritmo
        res = list(algoritmo(funcao, w_i, w_m, w_s, t_mut, t_com, dim=dim))
        print(f"Executou o algoritmo pela {ex+1}ª vez...")

        # Coloca o melhor de cada geração do algoritmo em uma lista
        for gen in range(len(res)):
            bests.append([res[gen][0], res[gen][1]])
            # print(f"{i}. [{result[i][0]}, {result[i][1]}] (taxa de melhora: {result[i][2]}%)")

        # Coloca o melhor de cada geração daquela execução na lista dos melhores de todas as execuções
        all_bests.append(bests)

        # log_hc = []
        # for i in range(len(res_deepso_hc)):
        #     if res_deepso_hc[i][2] != "":
        #         log_hc.append(res_deepso_hc[i][2])
        # all_logs_hc.append(log_hc)

    melhores_inds_por_gen = []
    melhores_valores_por_gen = []
    # log_hc_melhora_por_gen = []

    i = 0
    for j in range(len(all_bests[i])):
        melhores_inds_por_gen.append([])
        melhores_valores_por_gen.append([])

        for i in range(len(all_bests)):
            melhores_inds_por_gen[j].append(all_bests[i][j][0])
            melhores_valores_por_gen[j].append(all_bests[i][j][1])

    # i = 0
    # for j in range(len(all_logs_hc[i])):
    #     log_hc_melhora_por_gen.append([])
    #     for i in range(len(all_logs_hc)):
    #         log_hc_melhora_por_gen[j].append(all_logs_hc[i][j])

    medias = []
    log_hc_medias = []

    for i in range(len(melhores_valores_por_gen)):
        medias.append(sum(melhores_valores_por_gen[i])/len(melhores_valores_por_gen[i]))

    # for i in range(len(log_hc_melhora_por_gen)):
    #     log_hc_medias.append(sum(log_hc_melhora_por_gen[i])/len(log_hc_melhora_por_gen[i]))

    melhores_inds_ult_gen_ordenados_por_fitness = sort_list(melhores_inds_por_gen[-1], melhores_valores_por_gen[-1])

    return medias, melhores_valores_por_gen, melhores_inds_ult_gen_ordenados_por_fitness

def analise_geral(exs, teste=False):
    print("RODANDO DEEPSO 10 DIMENSÕES (ROSENBROCK)")
    deepso10_ros_medias, deepso10_ros_melhores_valores_por_gen, deepso10_ros_melhores_inds_ult_gen_ordenados_por_fitness = analise(exs, meu_deepso, rosenbrock_n,
                                                                                            w_i=0.53, w_m=0.2, w_s=0.17, t_mut=0.0027, t_com=0.959, dim=10)
    if teste == False:
        print("RODANDO DEEPSO 30 DIMENSÕES (ROSENBROCK)")
        deepso30_ros_medias, deepso30_ros_melhores_valores_por_gen, deepso30_ros_melhores_inds_ult_gen_ordenados_por_fitness = analise(exs, meu_deepso, rosenbrock_n,
                                                                                                w_i=0.53, w_m=0.2, w_s=0.17, t_mut=0.0027, t_com=0.959, dim=30)
        
        print("RODANDO DEEPSO 50 DIMENSÕES (ROSENBROCK)")
        deepso50_ros_medias, deepso50_ros_melhores_valores_por_gen, deepso50_ros_melhores_inds_ult_gen_ordenados_por_fitness = analise(exs, meu_deepso, rosenbrock_n,
                                                                                                w_i=0.53, w_m=0.2, w_s=0.17, t_mut=0.0027, t_com=0.959, dim=50)
    
    print("RODANDO C-DEEPSO 10 DIMENSÕES (ROSENBROCK)")
    cdeepso10_ros_medias, cdeepso10_ros_melhores_valores_por_gen, cdeepso10_ros_melhores_inds_ult_gen_ordenados_por_fitness = analise(exs, meu_deepso, rosenbrock_n,
                                                                                            w_i=0.53, w_m=0.2, w_s=0.17, t_mut=0.0027, t_com=0.959, dim=10)
    if teste == False:
        print("RODANDO C-DEEPSO 30 DIMENSÕES (ROSENBROCK)")
        cdeepso30_ros_medias, cdeepso30_ros_melhores_valores_por_gen, cdeepso30_ros_melhores_inds_ult_gen_ordenados_por_fitness = analise(exs, meu_deepso, rosenbrock_n,
                                                                                                w_i=0.53, w_m=0.2, w_s=0.17, t_mut=0.0027, t_com=0.959, dim=30)
        
        print("RODANDO C-DEEPSO 50 DIMENSÕES (ROSENBROCK)")
        cdeepso50_ros_medias, cdeepso50_ros_melhores_valores_por_gen, cdeepso50_ros_melhores_inds_ult_gen_ordenados_por_fitness = analise(exs, meu_deepso, rosenbrock_n,
                                                                                            w_i=0.53, w_m=0.2, w_s=0.17, t_mut=0.0027, t_com=0.959, dim=50)
        
        print("RODANDO DEEPSO 10 DIMENSÕES (RASTRIGIN)")
        deepso10_ras_medias, deepso10_ras_melhores_valores_por_gen, deepso10_ras_melhores_inds_ult_gen_ordenados_por_fitness = analise(exs, meu_deepso, rastrigin_n,
                                                                                                w_i=0.53, w_m=0.2, w_s=0.17, t_mut=0.0027, t_com=0.959, dim=10)

        print("RODANDO DEEPSO 30 DIMENSÕES (RASTRIGIN)")
        deepso30_ras_medias, deepso30_ras_melhores_valores_por_gen, deepso30_ras_melhores_inds_ult_gen_ordenados_por_fitness = analise(exs, meu_deepso, rastrigin_n,
                                                                                                w_i=0.53, w_m=0.2, w_s=0.17, t_mut=0.0027, t_com=0.959, dim=30)
        
        print("RODANDO DEEPSO 50 DIMENSÕES (RASTRIGIN)")
        deepso50_ras_medias, deepso50_ras_melhores_valores_por_gen, deepso50_ras_melhores_inds_ult_gen_ordenados_por_fitness = analise(exs, meu_deepso, rastrigin_n,
                                                                                                w_i=0.53, w_m=0.2, w_s=0.17, t_mut=0.0027, t_com=0.959, dim=50)
        
        print("RODANDO C-DEEPSO 10 DIMENSÕES (RASTRIGIN)")
        cdeepso10_ras_medias, cdeepso10_ras_melhores_valores_por_gen, cdeepso10_ras_melhores_inds_ult_gen_ordenados_por_fitness = analise(exs, meu_deepso, rastrigin_n,
                                                                                                w_i=0.53, w_m=0.2, w_s=0.17, t_mut=0.0027, t_com=0.959, dim=10)
        
        print("RODANDO C-DEEPSO 30 DIMENSÕES (RASTRIGIN)")
        cdeepso30_ras_medias, cdeepso30_ras_melhores_valores_por_gen, cdeepso30_ras_melhores_inds_ult_gen_ordenados_por_fitness = analise(exs, meu_deepso, rastrigin_n,
                                                                                                w_i=0.53, w_m=0.2, w_s=0.17, t_mut=0.0027, t_com=0.959, dim=30)
        
        print("RODANDO C-DEEPSO 50 DIMENSÕES (RASTRIGIN)")
        cdeepso50_ras_medias, cdeepso50_ras_melhores_valores_por_gen, cdeepso50_ras_melhores_inds_ult_gen_ordenados_por_fitness = analise(exs, meu_deepso, rastrigin_n,
                                                                                                w_i=0.53, w_m=0.2, w_s=0.17, t_mut=0.0027, t_com=0.959, dim=50)
        
        # Rosenbrock 10 dimensões
        plota_algoritmos(deepso10_ros_medias, cdeepso10_ros_medias, deepso10_ros_melhores_valores_por_gen, cdeepso10_ros_melhores_valores_por_gen, 
                        deepso10_ros_melhores_inds_ult_gen_ordenados_por_fitness, cdeepso10_ros_melhores_inds_ult_gen_ordenados_por_fitness, "Rosenbrock 10 dimensões")

        # Rosenbrock 30 dimensões
        plota_algoritmos(deepso30_ros_medias, cdeepso30_ros_medias, deepso30_ros_melhores_valores_por_gen, cdeepso30_ros_melhores_valores_por_gen, 
                        deepso30_ros_melhores_inds_ult_gen_ordenados_por_fitness, cdeepso30_ros_melhores_inds_ult_gen_ordenados_por_fitness, "Rosenbrock 30 dimensões")

        # Rosenbrock 50 dimensões
        plota_algoritmos(deepso50_ros_medias, cdeepso50_ros_medias, deepso50_ros_melhores_valores_por_gen, cdeepso50_ros_melhores_valores_por_gen, 
                        deepso50_ros_melhores_inds_ult_gen_ordenados_por_fitness, cdeepso50_ros_melhores_inds_ult_gen_ordenados_por_fitness, "Rosenbrock 50 dimensões")

        # Rastrigin 10 dimensões
        plota_algoritmos(deepso10_ras_medias, cdeepso10_ras_medias, deepso10_ras_melhores_valores_por_gen, cdeepso10_ras_melhores_valores_por_gen, 
                        deepso10_ras_melhores_inds_ult_gen_ordenados_por_fitness, cdeepso10_ras_melhores_inds_ult_gen_ordenados_por_fitness, "Rastrigin 10 dimensões")

        # Rastrigin 30 dimensões
        plota_algoritmos(deepso30_ras_medias, cdeepso30_ras_medias, deepso30_ras_melhores_valores_por_gen, cdeepso30_ras_melhores_valores_por_gen, 
                        deepso30_ras_melhores_inds_ult_gen_ordenados_por_fitness, cdeepso30_ras_melhores_inds_ult_gen_ordenados_por_fitness, "Rastrigin 30 dimensões")

        # Rastrigin 50 dimensões
        plota_algoritmos(deepso50_ras_medias, cdeepso50_ras_medias, deepso50_ras_melhores_valores_por_gen, cdeepso50_ras_melhores_valores_por_gen, 
                        deepso50_ras_melhores_inds_ult_gen_ordenados_por_fitness, cdeepso50_ras_melhores_inds_ult_gen_ordenados_por_fitness, "Rastrigin 50 dimensões")

        return [deepso10_ros_melhores_inds_ult_gen_ordenados_por_fitness, cdeepso10_ros_melhores_inds_ult_gen_ordenados_por_fitness,
                deepso30_ros_melhores_inds_ult_gen_ordenados_por_fitness, cdeepso30_ros_melhores_inds_ult_gen_ordenados_por_fitness,
                deepso50_ros_melhores_inds_ult_gen_ordenados_por_fitness, cdeepso50_ros_melhores_inds_ult_gen_ordenados_por_fitness,
                deepso10_ras_melhores_inds_ult_gen_ordenados_por_fitness, cdeepso10_ras_melhores_inds_ult_gen_ordenados_por_fitness,
                deepso30_ras_melhores_inds_ult_gen_ordenados_por_fitness, cdeepso30_ras_melhores_inds_ult_gen_ordenados_por_fitness,
                deepso50_ras_melhores_inds_ult_gen_ordenados_por_fitness, cdeepso50_ras_melhores_inds_ult_gen_ordenados_por_fitness]
    else:
        return deepso10_ros_melhores_inds_ult_gen_ordenados_por_fitness, cdeepso10_ros_melhores_inds_ult_gen_ordenados_por_fitness

def plota_algoritmos(medias1, medias2, melhores_valores_por_gen1, melhores_valores_por_gen2, 
                     melhores_inds_ult_gen_ordenados_por_fitness1, melhores_inds_ult_gen_ordenados_por_fitness2, titulo):
    plt.figure(figsize=(10, 6))
    plt.plot(list(range(0, 100)), medias1, linestyle='solid', label='DEEPSO')
    plt.plot(list(range(0, 100)), medias2, linestyle='dashed', label='C-DEEPSO')
    
    plt.title(titulo)
    plt.legend()
    plt.xlabel('Geração')
    plt.ylabel('Média')
    plt.yscale('log')

    plt.show()

    print()
    print("Dados sobre os melhores indivíduos nas últimas gerações das 30 execuções do deepso:")
    print(f"Média: {medias1[-1]}")
    print(f"Desvio Padrão: {np.std(melhores_valores_por_gen1[-1])}")
    print(f"Mediana: {np.median(melhores_valores_por_gen1[-1])}")
    print(f"Indivíduos que representam a Mediana: {melhores_inds_ult_gen_ordenados_por_fitness1[14]}, {melhores_inds_ult_gen_ordenados_por_fitness1[15]}")
    print("Como temos 30 execuções (número par), o valor da mediana corresponde à média do fitness dos dois pontos acima")

    print("\nDados sobre os melhores indivíduos nas últimas gerações das 30 execuções do deepso_hc:")
    print(f"Média: {medias2[-1]}")
    print(f"Desvio Padrão: {np.std(melhores_valores_por_gen2[-1])}")
    print(f"Mediana: {np.median(melhores_valores_por_gen2[-1])}")
    print(f"Indivíduos que representam a Mediana: {melhores_inds_ult_gen_ordenados_por_fitness2[14]}, {melhores_inds_ult_gen_ordenados_por_fitness2[15]}")
    print("Como temos 30 execuções (número par), o valor da mediana corresponde à média do fitness dos dois pontos acima")

    # # Análise do HC
    # plt.figure(figsize=(10, 6))
    # plt.plot(list(range(20, 30)), log_hc_medias, linestyle='solid', label='Melhora')

    # plt.title("Melhora de Xgb proporcionada pelo Hill Climbing no DEEPSO entre as gerações 20 e 30")
    # plt.legend()
    # plt.xlabel('Geração')
    # plt.ylabel('Melhora do Hill Climbing')
    # plt.yscale('log')

    # plt.show()

    # print(log_hc_medias)

def boxplot(data, titulo):
    bp1 = []
    bp2 = []
    for i in range(len(data[0])):
        bp1.append(rosenbrock_n(data[0][i], len(data[0][i])))
        bp2.append(rosenbrock_n(data[1][i], len(data[1][i])))
    
    plt.boxplot([bp1, bp2])
    labels = ['DEEPSO', 'C-DEEPSO']

    plt.title(titulo)
    plt.xlabel('Category')
    plt.ylabel('Values')
    plt.xticks([1, 2], labels)
    plt.show()

# 'w_i': 0.49407190293404785, 'w_m': 0.5110297319183723, 'w_s': 0.8376070505693844, 't_mut': 0.058184368025790606, 't_com': 0.12113239780216252
# roda(meu_deepso, rosenbrock_n, w_i=0.49407190293404785, w_m=0.5110297319183723,
#             w_s=0.8376070505693844, t_mut=0.058184368025790606, t_com=0.12113239780216252,
#             dim=10, limites=[-5, 5], num_particulas=30, max_iter=100,
#             com_hill_climb=False, plotar=False, printa_tudo=True)

roda(meu_cdeepso, rosenbrock_n, w_i=0.1936384307793284, w_m=0.3640416913945523, w_s=0.4423198778261193, t_mut=0.32984172878949947, t_com=0.5417890328667972,
            dim=10, limites=[-5, 5], num_particulas=30, max_iter=100,
            com_hill_climb=False, plotar=False, printa_tudo=True)

# inds_bp1, inds_bp2 = analise_geral(30, teste=True)
# boxplot([inds_bp1, inds_bp2], "Rosenbrock com 10 dimensões")

# encontra_variaveis(meu_deepso, rosenbrock_n, 10, 30, "DEEPSO Rosenbrock 10 dimensões")
# encontra_variaveis(meu_deepso, rosenbrock_n, 30, 30, "DEEPSO Rosenbrock 30 dimensões")
# encontra_variaveis(meu_deepso, rosenbrock_n, 50, 30, "DEEPSO Rosenbrock 50 dimensões")
# encontra_variaveis(meu_cdeepso, rosenbrock_n, 10, 30, "C-DEEPSO Rosenbrock 10 dimensões")
# encontra_variaveis(meu_cdeepso, rosenbrock_n, 30, 30, "C-DEEPSO Rosenbrock 30 dimensões")
# encontra_variaveis(meu_cdeepso, rosenbrock_n, 50, 30, "C-DEEPSO Rosenbrock 50 dimensões")

# encontra_variaveis(meu_deepso, rastrigin_n, 10, 30, "DEEPSO Rastrigin 10 dimensões")
# encontra_variaveis(meu_deepso, rastrigin_n, 30, 30, "DEEPSO Rastrigin 30 dimensões")
# encontra_variaveis(meu_deepso, rastrigin_n, 50, 30, "DEEPSO Rastrigin 50 dimensões")
# encontra_variaveis(meu_cdeepso, rastrigin_n, 10, 30, "C-DEEPSO Rastrigin 10 dimensões")
# encontra_variaveis(meu_cdeepso, rastrigin_n, 30, 30, "C-DEEPSO Rastrigin 30 dimensões")
# encontra_variaveis(meu_cdeepso, rastrigin_n, 50, 30, "C-DEEPSO Rastrigin 50 dimensões")