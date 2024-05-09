from genetico import rosenbrock
from genetico import alg_genetico
import matplotlib.pyplot as plt

my_results = []

for i in range(30):
    melhor, score, melhores = alg_genetico(rosenbrock, [[-2.048, 2.048], [-2.048, 2.048]], 16, 100, 100, 0.8, 0.03, 20, 2, False, False)
    my_results.append(melhores)
    print(f"{i+1}. Pior: {max(melhores)}")

my_melhores_por_gen = []
i = 0
for j in range(len(my_results[i])):
    my_melhores_por_gen.append([])
    for i in range(len(my_results)):
        my_melhores_por_gen[j].append(my_results[i][j])

print(f"Meus melhores: {my_melhores_por_gen}")

my_medias = []

for i in range(len(my_melhores_por_gen)):
    my_medias.append(sum(my_melhores_por_gen[i])/len(my_melhores_por_gen[i]))

for i in range(len(my_medias)):
    print(f"Geração {i+1}: {my_medias[i]}")

plt.plot(list(range(0, 100)), my_medias, label='Meu AG')
plt.legend()
plt.xlabel('Índice')
plt.ylabel('Média')

plt.show()