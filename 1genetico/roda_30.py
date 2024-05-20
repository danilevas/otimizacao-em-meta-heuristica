import genetico
import time
import genetico_gpt

limites = [[-2.048, 2.048], [-2.048, 2.048]]
geracoes = 100
n_bits = 16
tam_pop = 100
tx_cross = 0.8
tx_mut = 0.03
n_filhos = 20

results1 = []
results2 = []
results3 = []
results4 = []
results_gpt = []

for i in range(30):
    start = time.time()
    melhor, score, bests = genetico.alg_genetico(genetico.rosenbrock, limites, n_bits, geracoes, tam_pop, tx_cross, tx_mut, n_filhos, 1, False, False)
    end = time.time()
    print(f"Execução RD1 {i+1}: Melhor indivíduo: {genetico.decodifica(limites, n_bits, melhor)} = {score}\n({end-start}s)")
    results1.append(score)

for i in range(30):
    start = time.time()
    melhor, score, bests = genetico.alg_genetico(genetico.rosenbrock, limites, n_bits, geracoes, tam_pop, tx_cross, tx_mut, n_filhos, 2, False, False)
    end = time.time()
    print(f"Execução RD2 {i+1}: Melhor indivíduo: {genetico.decodifica(limites, n_bits, melhor)} = {score}\n({end-start}s)")
    results2.append(score)

for i in range(30):
    start = time.time()
    melhor, score, bests = genetico.alg_genetico(genetico.rosenbrock, limites, n_bits, geracoes, tam_pop, tx_cross, tx_mut, n_filhos, 3, False, False)
    end = time.time()
    print(f"Execução RD3 {i+1}: Melhor indivíduo: {genetico.decodifica(limites, n_bits, melhor)} = {score}\n({end-start}s)")
    results3.append(score)

for i in range(30):
    start = time.time()
    melhor, score, bests = genetico.alg_genetico(genetico.rosenbrock, limites, n_bits, geracoes, tam_pop, tx_cross, tx_mut, n_filhos, 4, False, False)
    end = time.time()
    print(f"Execução RD4 {i+1}: Melhor indivíduo: {genetico.decodifica(limites, n_bits, melhor)} = {score}\n({end-start}s)")
    results4.append(score)

for i in range(30):
    start = time.time()
    best_solution = genetico_gpt.genetico_gpt(tam_pop, (-2.048, 2.048), geracoes, tx_mut)
    end = time.time()
    print(f"Execução GPT {i+1}: Melhor indivíduo: {best_solution} = {genetico.rosenbrock(best_solution)}\n({end-start}s)")
    results_gpt.append(genetico.rosenbrock(best_solution))

print(f"Média 30 execuções com método roleta_dupla: {sum(results1)/len(results1)}")
print(f"Média 30 execuções com método roleta_dupla2: {sum(results2)/len(results2)}")
print(f"Média 30 execuções com método roleta_dupla3: {sum(results3)/len(results3)}")
print(f"Média 30 execuções com método roleta_dupla4: {sum(results4)/len(results4)}")
print(f"Média 30 execuções com método GPT: {sum(results_gpt)/len(results_gpt)}")