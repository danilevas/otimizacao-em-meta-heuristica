import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
from final import *
import time
from scipy.stats import ttest_ind_from_stats

def roda(algoritmo, funcao, w_i, w_m, w_s, t_mut, t_com, F, dim, limites, num_particulas, max_iter, com_hill_climb, plotar, printa_tudo, opcao):
    if funcao == meu_deepso:
        res_meu_deepso = list(algoritmo(funcao, w_i, w_m, w_s, t_mut, t_com, com_hill_climb, dim, limites, num_particulas, max_iter, plotar))
    else:
        res_meu_deepso = list(algoritmo(funcao, w_i, w_m, w_s, t_mut, t_com, F, com_hill_climb, dim, limites, num_particulas, max_iter, plotar, opcao))
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

def analise(exs, algoritmo, funcao, w_i, w_m, w_s, t_mut, t_com, F, num_particulas, dim):
    # Análise dos algoritmos
    # Executamos todos 30 vezes com 100 gerações
    bounds_rosenbrock = [(-5., 5.)]
    bounds_rastrigin = [(-5.12, 5.12)]

    all_bests = []
    all_logs_hc = []

    for ex in range(exs):
        bests = []

        # Roda o algoritmo
        if algoritmo == meu_deepso: res = list(algoritmo(funcao, w_i, w_m, w_s, t_mut, t_com, dim=dim, num_particulas=num_particulas, com_hill_climb=True))
        else: res = list(algoritmo(funcao, w_i, w_m, w_s, t_mut, t_com, F, dim=dim, num_particulas=num_particulas, com_hill_climb=True))
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

def analise_geral(exs, amostra="todos"):
    if amostra in ["ros_10", "todos"]:
        print("RODANDO DEEPSO 10 DIMENSÕES (ROSENBROCK)")
        deepso10_ros_medias, deepso10_ros_melhores_valores_por_gen, deepso10_ros_melhores_inds_ult_gen_ordenados_por_fitness = analise(exs, meu_deepso, rosenbrock_n,
                                                                                            w_i=0.5076009271957435, w_m=0.4429945932782201, w_s=0.3947439471627401,
                                                                                            t_mut=0.07267040486682608, t_com=0.19351372963410257, F=0,
                                                                                            num_particulas=86, dim=10)

        print("RODANDO C-DEEPSO 10 DIMENSÕES (ROSENBROCK)")
        cdeepso10_ros_medias, cdeepso10_ros_melhores_valores_por_gen, cdeepso10_ros_melhores_inds_ult_gen_ordenados_por_fitness = analise(exs, meu_deepso, rosenbrock_n,
                                                                                            w_i=0.21002922456374457, w_m=0.03894880433457305, w_s=0.4492438362995592,
                                                                                            t_mut=0.019436069163957098, t_com=0.5922906033349694, F=1.1025041161386335,
                                                                                            num_particulas=27, dim=10)
    if amostra in ["ros_30", "todos"]:
        print("RODANDO DEEPSO 30 DIMENSÕES (ROSENBROCK)")
        deepso30_ros_medias, deepso30_ros_melhores_valores_por_gen, deepso30_ros_melhores_inds_ult_gen_ordenados_por_fitness = analise(exs, meu_deepso, rosenbrock_n,
                                                                                                w_i=0.12229881979204243, w_m=0.5473388781280298, w_s=0.2738314472884503,
                                                                                                t_mut=0.5541076250130184, t_com=0.9115779947456945, F=0,
                                                                                                num_particulas=70,  dim=30)
        
        print("RODANDO C-DEEPSO 30 DIMENSÕES (ROSENBROCK)")
        cdeepso30_ros_medias, cdeepso30_ros_melhores_valores_por_gen, cdeepso30_ros_melhores_inds_ult_gen_ordenados_por_fitness = analise(exs, meu_deepso, rosenbrock_n,
                                                                                                w_i=0.11904852438917952, w_m=0.3310351478310495, w_s=0.10885958961331216,
                                                                                                t_mut=0.049673492983309274, t_com=0.8539183380313524, F=0.01384373412967433,
                                                                                                num_particulas=100,  dim=30)
    
    if amostra in ["ros_50", "todos"]:
        print("RODANDO DEEPSO 50 DIMENSÕES (ROSENBROCK)")
        deepso50_ros_medias, deepso50_ros_melhores_valores_por_gen, deepso50_ros_melhores_inds_ult_gen_ordenados_por_fitness = analise(exs, meu_deepso, rosenbrock_n,
                                                                                                w_i=0.018961926239525237, w_m=0.32107489666409283, w_s=0.715655791383361,
                                                                                                t_mut=0.3940003741978141, t_com=0.9543125369726769, F=0,
                                                                                                num_particulas=71,  dim=50)
        
        print("RODANDO C-DEEPSO 50 DIMENSÕES (ROSENBROCK)")
        cdeepso50_ros_medias, cdeepso50_ros_melhores_valores_por_gen, cdeepso50_ros_melhores_inds_ult_gen_ordenados_por_fitness = analise(exs, meu_deepso, rosenbrock_n,
                                                                                                w_i=0.3738701953900034, w_m=0.2684850678490869, w_s=0.01263204158929801,
                                                                                                t_mut=0.22112719771668227, t_com=0.11106313254757884, F=0.3698309341860955,
                                                                                                num_particulas=96,   dim=50)
    
    if amostra in ["ras_10", "todos"]:
        print("RODANDO DEEPSO 10 DIMENSÕES (RASTRIGIN)")
        deepso10_ras_medias, deepso10_ras_melhores_valores_por_gen, deepso10_ras_melhores_inds_ult_gen_ordenados_por_fitness = analise(exs, meu_deepso, rastrigin_n,
                                                                                                w_i=0.20418811781971788, w_m=0.5431065480893994, w_s=0.4626481233438281,
                                                                                                t_mut=0.08256039924070052, t_com=0.13938588490530093, F=0,
                                                                                                num_particulas=63,  dim=10)

        print("RODANDO C-DEEPSO 10 DIMENSÕES (RASTRIGIN)")
        cdeepso10_ras_medias, cdeepso10_ras_melhores_valores_por_gen, cdeepso10_ras_melhores_inds_ult_gen_ordenados_por_fitness = analise(exs, meu_deepso, rastrigin_n,
                                                                                                w_i=0.3755082678452043, w_m=0.07823468990809129, w_s=0.2310259757822647,
                                                                                                t_mut=0.03131878729340136, t_com=0.30063033310105064, F=0.35114252207517493,
                                                                                                num_particulas=60,  dim=10)

    if amostra in ["ras_30", "todos"]:
        print("RODANDO DEEPSO 30 DIMENSÕES (RASTRIGIN)")
        deepso30_ras_medias, deepso30_ras_melhores_valores_por_gen, deepso30_ras_melhores_inds_ult_gen_ordenados_por_fitness = analise(exs, meu_deepso, rastrigin_n,
                                                                                                w_i=0.05647555136854313, w_m=0.899490073868074, w_s=0.13946690260760802,
                                                                                                t_mut=0.659605898137305, t_com=0.9461297689025504, F=0,
                                                                                                num_particulas=70,  dim=30)
        
        print("RODANDO C-DEEPSO 30 DIMENSÕES (RASTRIGIN)")
        cdeepso30_ras_medias, cdeepso30_ras_melhores_valores_por_gen, cdeepso30_ras_melhores_inds_ult_gen_ordenados_por_fitness = analise(exs, meu_deepso, rastrigin_n,
                                                                                                w_i=0.3179615128927915, w_m=0.15052579505157043, w_s=0.3974876216221927,
                                                                                                t_mut=0.07407692387870876, t_com=0.1930975569008346, F=0.3824789579043441,
                                                                                                num_particulas=70,  dim=30)

    if amostra in ["ras_50", "todos"]:
        print("RODANDO DEEPSO 50 DIMENSÕES (RASTRIGIN)")
        deepso50_ras_medias, deepso50_ras_melhores_valores_por_gen, deepso50_ras_melhores_inds_ult_gen_ordenados_por_fitness = analise(exs, meu_deepso, rastrigin_n,
                                                                                                w_i=0.2866722221604476, w_m=0.8337307604335625, w_s=0.496486003831572,
                                                                                                t_mut=0.15270541720007136, t_com=0.8070367765225529, F=0,
                                                                                                num_particulas=50,  dim=50)
        
        print("RODANDO C-DEEPSO 50 DIMENSÕES (RASTRIGIN)")
        cdeepso50_ras_medias, cdeepso50_ras_melhores_valores_por_gen, cdeepso50_ras_melhores_inds_ult_gen_ordenados_por_fitness = analise(exs, meu_deepso, rastrigin_n,
                                                                                                w_i=0.0725609039905108, w_m=0.14735959223222378, w_s=0.42539120954474685,
                                                                                                t_mut=0.04489736152356841, t_com=0.6751728914348668, F=0.5859934595779772,
                                                                                                num_particulas=62,  dim=50)
    
    # Resetando dados.txt
    with open("5final/txts/dados.txt", "w", encoding="utf-8") as f:
        pass

    if amostra in ["ros_10", "todos"]:
        # Rosenbrock com 10 dimensões
        plota_algoritmos(deepso10_ros_medias, cdeepso10_ros_medias, deepso10_ros_melhores_valores_por_gen, cdeepso10_ros_melhores_valores_por_gen, 
                        deepso10_ros_melhores_inds_ult_gen_ordenados_por_fitness, cdeepso10_ros_melhores_inds_ult_gen_ordenados_por_fitness, "Rosenbrock com 10 dimensões")

        boxplot([deepso10_ros_melhores_inds_ult_gen_ordenados_por_fitness, cdeepso10_ros_melhores_inds_ult_gen_ordenados_por_fitness], rosenbrock_n, "Rosenbrock com 10 dimensões")
        t_testa(deepso10_ros_melhores_inds_ult_gen_ordenados_por_fitness, cdeepso10_ros_melhores_inds_ult_gen_ordenados_por_fitness)

    if amostra in ["ros_30", "todos"]:
        # Rosenbrock com 30 dimensões
        plota_algoritmos(deepso30_ros_medias, cdeepso30_ros_medias, deepso30_ros_melhores_valores_por_gen, cdeepso30_ros_melhores_valores_por_gen, 
                        deepso30_ros_melhores_inds_ult_gen_ordenados_por_fitness, cdeepso30_ros_melhores_inds_ult_gen_ordenados_por_fitness, "Rosenbrock com 30 dimensões")

        boxplot([deepso30_ros_melhores_inds_ult_gen_ordenados_por_fitness, cdeepso30_ros_melhores_inds_ult_gen_ordenados_por_fitness], rosenbrock_n, "Rosenbrock com 30 dimensões")
        t_testa(deepso30_ros_melhores_inds_ult_gen_ordenados_por_fitness, cdeepso30_ros_melhores_inds_ult_gen_ordenados_por_fitness)

    if amostra in ["ros_50", "todos"]:
        # Rosenbrock com 50 dimensões
        plota_algoritmos(deepso50_ros_medias, cdeepso50_ros_medias, deepso50_ros_melhores_valores_por_gen, cdeepso50_ros_melhores_valores_por_gen, 
                        deepso50_ros_melhores_inds_ult_gen_ordenados_por_fitness, cdeepso50_ros_melhores_inds_ult_gen_ordenados_por_fitness, "Rosenbrock com 50 dimensões")

        boxplot([deepso50_ros_melhores_inds_ult_gen_ordenados_por_fitness, cdeepso50_ros_melhores_inds_ult_gen_ordenados_por_fitness], rosenbrock_n, "Rosenbrock com 50 dimensões")
        t_testa(deepso50_ros_melhores_inds_ult_gen_ordenados_por_fitness, cdeepso50_ros_melhores_inds_ult_gen_ordenados_por_fitness)

    if amostra in ["ras_10", "todos"]:
        # Rastrigin com 10 dimensões
        plota_algoritmos(deepso10_ras_medias, cdeepso10_ras_medias, deepso10_ras_melhores_valores_por_gen, cdeepso10_ras_melhores_valores_por_gen, 
                        deepso10_ras_melhores_inds_ult_gen_ordenados_por_fitness, cdeepso10_ras_melhores_inds_ult_gen_ordenados_por_fitness, "Rastrigin com 10 dimensões")

        boxplot([deepso10_ras_melhores_inds_ult_gen_ordenados_por_fitness, cdeepso10_ras_melhores_inds_ult_gen_ordenados_por_fitness], rastrigin_n, "Rastrigin com 10 dimensões")
        t_testa(deepso10_ras_melhores_inds_ult_gen_ordenados_por_fitness, cdeepso10_ras_melhores_inds_ult_gen_ordenados_por_fitness)
        
    if amostra in ["ras_30", "todos"]:
        # Rastrigin com 30 dimensões
        plota_algoritmos(deepso30_ras_medias, cdeepso30_ras_medias, deepso30_ras_melhores_valores_por_gen, cdeepso30_ras_melhores_valores_por_gen, 
                        deepso30_ras_melhores_inds_ult_gen_ordenados_por_fitness, cdeepso30_ras_melhores_inds_ult_gen_ordenados_por_fitness, "Rastrigin com 30 dimensões")

        boxplot([deepso30_ras_melhores_inds_ult_gen_ordenados_por_fitness, cdeepso30_ras_melhores_inds_ult_gen_ordenados_por_fitness], rastrigin_n, "Rastrigin com 30 dimensões")
        t_testa(deepso30_ras_melhores_inds_ult_gen_ordenados_por_fitness, cdeepso30_ras_melhores_inds_ult_gen_ordenados_por_fitness)

    if amostra in ["ras_50", "todos"]:
        # Rastrigin com 50 dimensões
        plota_algoritmos(deepso50_ras_medias, cdeepso50_ras_medias, deepso50_ras_melhores_valores_por_gen, cdeepso50_ras_melhores_valores_por_gen, 
                        deepso50_ras_melhores_inds_ult_gen_ordenados_por_fitness, cdeepso50_ras_melhores_inds_ult_gen_ordenados_por_fitness, "Rastrigin com 50 dimensões")

        boxplot([deepso50_ras_melhores_inds_ult_gen_ordenados_por_fitness, cdeepso50_ras_melhores_inds_ult_gen_ordenados_por_fitness], rastrigin_n, "Rastrigin com 50 dimensões")
        t_testa(deepso50_ras_melhores_inds_ult_gen_ordenados_por_fitness, cdeepso50_ras_melhores_inds_ult_gen_ordenados_por_fitness)

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

    plt.savefig(f'5final/graficos/grafico_{titulo}.png')
    plt.show()

    with open("5final/txts/dados.txt", "a", encoding="utf-8") as f:
        f.write(f"{titulo}:\n")
        f.write("\nDados sobre os melhores indivíduos nas últimas gerações das 30 execuções do DEEPSO:\n")
        f.write(f"Média: {medias1[-1]}\n")
        f.write(f"Desvio Padrão: {np.std(melhores_valores_por_gen1[-1])}\n")
        f.write(f"Mediana: {np.median(melhores_valores_por_gen1[-1])}\n")
        f.write(f"Indivíduos que representam a Mediana: {melhores_inds_ult_gen_ordenados_por_fitness1[14]}, {melhores_inds_ult_gen_ordenados_por_fitness1[15]}\n")
        f.write("Como temos 30 execuções (número par), o valor da mediana corresponde à média do fitness dos dois pontos acima\n")

        f.write("\nDados sobre os melhores indivíduos nas últimas gerações das 30 execuções do C-DEEPSO:\n")
        f.write(f"Média: {medias2[-1]}\n")
        f.write(f"Desvio Padrão: {np.std(melhores_valores_por_gen2[-1])}\n")
        f.write(f"Mediana: {np.median(melhores_valores_por_gen2[-1])}\n")
        f.write(f"Indivíduos que representam a Mediana: {melhores_inds_ult_gen_ordenados_por_fitness2[14]}, {melhores_inds_ult_gen_ordenados_por_fitness2[15]}\n")
        f.write("Como temos 30 execuções (número par), o valor da mediana corresponde à média do fitness dos dois pontos acima\n")

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

def t_testa(melhores_30_deepso, melhores_30_cdeepso):
    # Calculate summary statistics for group1
    mean1 = np.mean(melhores_30_deepso)
    std1 = np.std(melhores_30_deepso, ddof=1)  # Use ddof=1 to get the sample standard deviation
    n1 = len(melhores_30_deepso)

    # Calculate summary statistics for group2
    mean2 = np.mean(melhores_30_cdeepso)
    std2 = np.std(melhores_30_cdeepso, ddof=1)
    n2 = len(melhores_30_cdeepso)

    # Perform the t-test using ttest_ind_from_stats
    t_stat, p_value = ttest_ind_from_stats(mean1, std1, n1, mean2, std2, n2)

    with open("5final/txts/dados.txt", "a", encoding="utf-8") as f:
        f.write(f"\nT-statistic: {t_stat}\n")
        f.write(f"P-value: {p_value}\n\n")

def boxplot(data, funcao, titulo):
    bp1 = []
    bp2 = []
    for i in range(len(data[0])):
        bp1.append(funcao(data[0][i], len(data[0][i])))
        bp2.append(funcao(data[1][i], len(data[1][i])))
    
    plt.boxplot([bp1, bp2])
    labels = ['DEEPSO', 'C-DEEPSO']

    plt.title(titulo)
    plt.xlabel('Category')
    plt.ylabel('Values')
    plt.yscale('log')
    plt.xticks([1, 2], labels)
    plt.savefig(f'5final/graficos/boxplot_{titulo}.png')
    plt.show()

# roda(meu_cdeepso, rosenbrock_n, w_i=0.21002922456374457, w_m=0.03894880433457305, w_s=0.4492438362995592, t_mut=0.019436069163957098, t_com=0.5922906033349694, F=1.1025041161386335, num_particulas=27,
#             dim=10, limites=[-5, 5], max_iter=100,
#             com_hill_climb=False, plotar=False, printa_tudo=True, opcao=1)

# roda(meu_cdeepso, rosenbrock_n, w_i=0.21002922456374457, w_m=0.03894880433457305, w_s=0.4492438362995592, t_mut=0.019436069163957098, t_com=0.5922906033349694, F=1.1025041161386335, num_particulas=27,
#             dim=10, limites=[-5, 5], max_iter=100,
#             com_hill_climb=False, plotar=False, printa_tudo=True, opcao=2)

start = time.time()
analise_geral(30, amostra="todos")
print(f"O código levou {time.time()-start} segundos")

# 'w_i': 0.49407190293404785, 'w_m': 0.5110297319183723, 'w_s': 0.8376070505693844, 't_mut': 0.058184368025790606, 't_com': 0.12113239780216252
# roda(meu_deepso, rosenbrock_n, w_i=0.49407190293404785, w_m=0.5110297319183723,
#             w_s=0.8376070505693844, t_mut=0.058184368025790606, t_com=0.12113239780216252, F=0,
#             dim=10, limites=[-5, 5], num_particulas=30, max_iter=100,
#             com_hill_climb=False, plotar=False, printa_tudo=True)

# roda(meu_cdeepso, rosenbrock_n, w_i=0.1936384307793284, w_m=0.3640416913945523, w_s=0.4423198778261193, t_mut=0.32984172878949947, t_com=0.5417890328667972, F=1.8,
#             dim=10, limites=[-5, 5], num_particulas=30, max_iter=100,
#             com_hill_climb=False, plotar=False, printa_tudo=True)

# inds_bp1, inds_bp2 = analise_geral(30, teste=True)
# boxplot([inds_bp1, inds_bp2], "Rosenbrock com 10 dimensões")

# encontra_variaveis(meu_deepso, rosenbrock_n, 10, 30, "DEEPSO Rosenbrock com 10 dimensões")
# encontra_variaveis(meu_deepso, rosenbrock_n, 30, 30, "DEEPSO Rosenbrock com 30 dimensões")
# encontra_variaveis(meu_deepso, rosenbrock_n, 50, 30, "DEEPSO Rosenbrock com 50 dimensões")
# encontra_variaveis(meu_cdeepso, rosenbrock_n, 10, 30, "C-DEEPSO Rosenbrock com 10 dimensões")
# encontra_variaveis(meu_cdeepso, rosenbrock_n, 30, 30, "C-DEEPSO Rosenbrock com 30 dimensões")
# encontra_variaveis(meu_cdeepso, rosenbrock_n, 50, 30, "C-DEEPSO Rosenbrock com 50 dimensões")

# encontra_variaveis(meu_deepso, rastrigin_n, 10, 30, "DEEPSO Rastrigin com 10 dimensões")
# encontra_variaveis(meu_deepso, rastrigin_n, 30, 30, "DEEPSO Rastrigin com 30 dimensões")
# encontra_variaveis(meu_deepso, rastrigin_n, 50, 30, "DEEPSO Rastrigin com 50 dimensões")
# encontra_variaveis(meu_cdeepso, rastrigin_n, 10, 30, "C-DEEPSO Rastrigin com 10 dimensões")
# encontra_variaveis(meu_cdeepso, rastrigin_n, 30, 30, "C-DEEPSO Rastrigin com 30 dimensões")
# encontra_variaveis(meu_cdeepso, rastrigin_n, 50, 30, "C-DEEPSO Rastrigin com 50 dimensões")