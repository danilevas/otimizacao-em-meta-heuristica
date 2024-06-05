# Solid
# Dashed
# Dotted
# Dashdot
# None

plt.figure(figsize=(10, 6))
plt.plot(list(range(0, 100)), meu_pso_medias, linestyle='solid', label='rand1')
plt.plot(list(range(0, 100)), best1_medias, linestyle='dashed', label='best1')
plt.plot(list(range(0, 100)), currenttobest1_medias, linestyle='dotted', label='currenttobest1')

plt.legend()
plt.xlabel('Geração')
plt.ylabel('Média')
plt.yscale('log')

plt.show()

print()
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