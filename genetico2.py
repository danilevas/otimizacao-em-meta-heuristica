from numpy.random import randint
from numpy.random import rand
import numpy.random as npr
import matplotlib.pyplot as plt
import numpy as np
import time
from plot import plot
from plot import plot_landscape

# Rosenbrock
def rosenbrock(X, a=1, b=100):
    x, y = X
    return (a - x)**2 + b * (y - x**2)**2
        
# Decodifica a bitstring em números
# Divide a bitstring em 2 metades de 16 bits cada
# Transforma cada um em um número base 10
# Faz uma regra de 3 entre ele e o maior possível com 16 bits, e coloca no range dos bounds
# Assim temos um x e um y no range correto
def decodifica(limites, n_bits, bitstring):
	decodificado = list()
	maior = 2**n_bits
	for i in range(len(limites)):
		# Extrai a substring
		start, end = i * n_bits, (i * n_bits)+n_bits
		substring = bitstring[start:end]
		# Converte a substring para uma string de chars
		chars = ''.join([str(s) for s in substring])
		# Converte string em int
		integer = int(chars, 2)
		# Coloca o int no range desejado
		valor = limites[i][0] + (integer/maior) * (limites[i][1] - limites[i][0])
		# Armazena
		decodificado.append(valor)
	# print(f"Resultado: {decodificado}")
	return decodificado

def pega_menor(scores):
    return scores.index(min(scores))

def pega_maior(scores):
    return scores.index(max(scores))

def roleta_dupla(pop, scores):
    scores_inv = [1/s for s in scores]
    max = sum(scores_inv)
    probs_selecao = [s/max for s in scores_inv]
    pai1 = pop[npr.choice(len(pop), p=probs_selecao)]
    pai2 = pop[npr.choice(len(pop), p=probs_selecao)]
    cont = 0
    while pai1 == pai2:
        cont += 1
        pai2 = pop[npr.choice(len(pop), p=probs_selecao)]
        # Se der o mesmo pai mais de 30 vezes, deixa ser o mesmo pai (reprodução assexual)
        if cont == 30: 
            print("Tive que sair do loop!")
            break
    return pai1, pai2

def roleta_dupla2(pop, scores):
    max = sum(scores)
    probs_selecao = [max-s for s in scores]
    new_max = sum(probs_selecao)
    probs_selecao = [p/new_max for p in probs_selecao]
    pai1 = pop[npr.choice(len(pop), p=probs_selecao)]
    pai2 = pop[npr.choice(len(pop), p=probs_selecao)]
    while pai1 == pai2:
        pai2 = pop[npr.choice(len(pop), p=probs_selecao)]
    return pai1, pai2

def roleta_dupla3(pop, scores):
    max = sum(scores)
    probs_selecao = []
    for i in range(len(scores)):
        if scores[i]*100 < max:
            probs_selecao.append(max-(scores[i]*100))
        else:
            probs_selecao.append(0)
    new_max = sum(probs_selecao)
    if new_max == 0:
        print("Todos iguais!")
        pai1 = pop[0]
        pai2 = pop[1]
        return pai1, pai2
    probs_selecao = [p/new_max for p in probs_selecao]
    pai1 = pop[npr.choice(len(pop), p=probs_selecao)]
    pai2 = pop[npr.choice(len(pop), p=probs_selecao)]
    cont = 0
    while pai1 == pai2:
        cont += 1
        pai2 = pop[npr.choice(len(pop), p=probs_selecao)]
        # Se der o mesmo pai mais de 30 vezes, deixa ser o mesmo pai (reprodução assexual)
        if cont == 30:
            print("Tive que sair do loop!")
            break
    return pai1, pai2

def roleta_dupla4(pop, scores):
    max = sum(scores)
    probs_selecao = []
    for i in range(len(scores)):
        if scores[i]*2 < max:
            probs_selecao.append(max-(scores[i]*2))
        else:
            probs_selecao.append(0)
    new_max = sum(probs_selecao)
    probs_selecao = [p/new_max for p in probs_selecao]
    pai1 = pop[npr.choice(len(pop), p=probs_selecao)]
    pai2 = pop[npr.choice(len(pop), p=probs_selecao)]
    cont = 0
    while pai1 == pai2:
        cont += 1
        pai2 = pop[npr.choice(len(pop), p=probs_selecao)]
        # Se der o mesmo pai mais de 30 vezes, deixa ser o mesmo pai (reprodução assexual)
        if cont == 30:
            print("Tive que sair do loop!")
            break
    return pai1, pai2

# Faz o crossover de dois pais para criar um filho
def crossover(p1, p2, tx_cross):
	# Por padrão, filhos são cópias dos pais (estamos gerando 2 gêmeos fraternos, para depois escolher 1)
	f1, f2 = p1.copy(), p2.copy()
	# Checa se irá recombinar
	if rand() < tx_cross:
		# Seleciona ponto de crossover que não seja no fim da string
		pt = randint(1, len(p1)-2)
		# Faz o crossover
		f1 = p1[:pt] + p2[pt:]
		f2 = p2[:pt] + p1[pt:]
	# Escolhe um dos filhos aleatoriamente
	if rand() < 0.5:
		return f1
	else:
		return f2

# Operador de mutação
def mutacao(bitstring, tx_mut):
	for i in range(len(bitstring)):
		# Checa por mutação
		if rand() < tx_mut:
			# Muda o bit
			bitstring[i] = 1 - bitstring[i]

# Algoritmo genético
def alg_genetico(funcao, limites, n_bits, geracoes, tam_pop, tx_cross, tx_mut, n_filhos, roleta, verbose, plotar):
	# População inicial de bitstrings aleatórias
	pop = [randint(0, 2, n_bits*len(limites)).tolist() for _ in range(tam_pop)]
	# Decodifica todos os itens
	decodificado = [decodifica(limites, n_bits, p) for p in pop]
	# Avalia todos os candidatos da população
	scores = [funcao(d) for d in decodificado]
	# Checa pela melhor heurística atual
	i_melhor = pega_menor(scores)
	melhor_eval = scores[i_melhor]
	if verbose: print(f"> Melhor inicial: {decodificado[i_melhor]} = {melhor_eval}")
	bests = []
	if plotar:
		print("Geração 1")
		plot(decodificado)
		plot_landscape(decodificado)

	# Iterando pelas gerações
	for gen in range(geracoes):
        for i in range(n_filhos):
            # Seleciona 2 pais com o método escolhido
            if roleta == 1: pai1, pai2 = roleta_dupla(pop, scores)
            elif roleta == 2: pai1, pai2 = roleta_dupla2(pop, scores)
            elif roleta == 3: pai1, pai2 = roleta_dupla3(pop, scores)
            elif roleta == 4: pai1, pai2 = roleta_dupla4(pop, scores)
            else:
                raise Exception("O método escolhido não existe")
            # Crossover
            filho = crossover(pai1, pai2, tx_cross)
            # Mutação
            mutacao(filho, tx_mut)
            # Substituímos o indivíduo de maior heurística pelo novo filho
            pop[pega_maior(scores)] = filho
            scores[pega_maior(scores)] = funcao(decodifica(limites, n_bits, filho))

		# Decodifica todos os itens
		decodificado = [decodifica(limites, n_bits, p) for p in pop]
		# Avalia todos os candidatos da população
		scores = [funcao(d) for d in decodificado]
		# Checa por uma nova melhor solução
		i_possivel_melhor = pega_menor(scores)
		possivel_melhor_eval = scores[i_melhor]
		# Se o melhor indivíduo daquela geração for melhor que o melhor histórico, substituímos o melhor histórico por ele
		if possivel_melhor_eval < melhor_eval:
			i_melhor = i_possivel_melhor
			melhor_eval = possivel_melhor_eval
			if verbose: print(f"> Geração {gen+1}: Novo melhor {decodificado[i_melhor]} = {melhor_eval}")

		# Plota nas gerações 1, 25, 50 e 100
		if plotar and (gen == 24 or gen == 49 or gen == 99):
			print(f"Geração {gen+1}")
			plot(decodificado)
			plot_landscape(decodificado)

		scores_em_ordem = scores.copy()
		scores_em_ordem = sorted(scores_em_ordem)
		# Colocando a média dos 30 melhores na lista de bests
		bests.append(sum(scores_em_ordem[:30])/30)

	return pop[i_melhor], melhor_eval, bests

# Range para input
limites = [[-2.048, 2.048], [-2.048, 2.048]]
# Total de gerações
geracoes = 100
# Bits por variável
n_bits = 16
# Tamanho da população
tam_pop = 100
# Taxa de crossover
tx_cross = 0.8
# Taxa de mutação
tx_mut = 0.03
# Número de filhos por geração
n_filhos = 20
# Faz a busca do algoritmo genético
melhor, score, bests = alg_genetico(rosenbrock, limites, n_bits, geracoes, tam_pop, tx_cross, tx_mut, n_filhos, 3, True, False)
print(f"Melhor indivíduo encontrado por mim: {decodifica(limites, n_bits, melhor)} = {score}")