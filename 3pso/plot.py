# Solid
# Dashed
# Dotted
# Dashdot
# None

import matplotlib as plt

plt.figure(figsize=(10, 6))
plt.plot(list(range(0, 100)), meu_pso_medias, linestyle='solid', label='meu_pso')
plt.plot(list(range(0, 100)), pymoo_medias, linestyle='dashed', label='pymoo')
plt.plot(list(range(0, 100)), pyswarm_medias, linestyle='dotted', label='pyswarm')

plt.legend()
plt.xlabel('Geração')
plt.ylabel('Média')
plt.yscale('log')

plt.show()

print()
print("Dados sobre os melhores indivíduos nas últimas gerações das 30 execuções do meu_pso:")
print(f"Média: {meu_pso_medias[-1]}")
print(f"Desvio Padrão: {np.std(meu_pso_melhores_valores_por_gen[-1])}")
print(f"Mediana: {np.median(meu_pso_melhores_valores_por_gen[-1])}")
print(f"Indivíduos que representam a Mediana: {meu_pso_melhores_inds_ult_gen_ordenados_por_fitness[14]}, {meu_pso_melhores_inds_ult_gen_ordenados_por_fitness[15]}")
print("Como temos 30 execuções (número par), o valor da mediana corresponde à média do fitness dos dois pontos acima")

print("\nDados sobre os melhores indivíduos nas últimas gerações das 30 execuções do pymoo:")
print(f"Média: {pymoo_medias[-1]}")
print(f"Desvio Padrão: {np.std(pymoo_melhores_valores_por_gen[-1])}")
print(f"Mediana: {np.median(pymoo_melhores_valores_por_gen[-1])}")
print(f"Indivíduos que representam a Mediana: {pymoo_melhores_inds_ult_gen_ordenados_por_fitness[14]}, {pymoo_melhores_inds_ult_gen_ordenados_por_fitness[15]}")
print("Como temos 30 execuções (número par), o valor da mediana corresponde à média do fitness dos dois pontos acima")

print("\nDados sobre os melhores indivíduos nas últimas gerações das 30 execuções do pyswarm:")
print(f"Média: {pyswarm_medias[-1]}")
print(f"Desvio Padrão: {np.std(pyswarm_melhores_valores_por_gen[-1])}")
print(f"Mediana: {np.median(pyswarm_melhores_valores_por_gen[-1])}")
print(f"Indivíduos que representam a Mediana: {pyswarm_melhores_inds_ult_gen_ordenados_por_fitness[14]}, {pyswarm_melhores_inds_ult_gen_ordenados_por_fitness[15]}")
print("Como temos 30 execuções (número par), o valor da mediana corresponde à média do fitness dos dois pontos acima")