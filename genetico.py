from numpy.random import randint
from numpy.random import rand
import numpy.random as npr

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
    while pai1 == pai2:
        pai2 = pop[npr.choice(len(pop), p=probs_selecao)]
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
def alg_genetico(rosenbrock, limites, n_bits, geracoes, tam_pop, tx_cross, tx_mut, n_filhos):
    # População inicial de bitstrings aleatórias
    pop = [randint(0, 2, n_bits*len(limites)).tolist() for _ in range(tam_pop)]
    # O melhor começa sendo o primeiro item e sua pontuação
    melhor, melhor_eval = 0, rosenbrock(decodifica(limites, n_bits, pop[0]))
    # Decodifica todos os itens
    decodificado = [decodifica(limites, n_bits, p) for p in pop]
    # Avalia todos os candidatos da população
    scores = [rosenbrock(d) for d in decodificado]
    # Checa pela menor heurística atual
    i_min = pega_menor(scores)
    print(f"> Melhor inicial: {decodificado[i_min]} = {scores[i_min]}")
    bests = []

    # Iterando pelas gerações
    for gen in range(geracoes):
        # if gen % 200 == 0 or gen == 999: print(f"Geração: {gen}")
        for i in range(n_filhos):
            # Seleciona 2 pais
            pai1, pai2 = roleta_dupla2(pop, scores)
            # Crossover
            filho = crossover(pai1, pai2, tx_cross)
            # Mutação
            mutacao(filho, tx_mut)
            # Substituímos o indivíduo de maior heurística pelo novo filho
            pop[pega_maior(scores)] = filho
            scores[pega_maior(scores)] = rosenbrock(decodifica(limites, n_bits, filho))
        
        # Decodifica todos os itens
        decodificado = [decodifica(limites, n_bits, p) for p in pop]
        # Avalia todos os candidatos da população
        scores = [rosenbrock(d) for d in decodificado]
        # Checa por uma nova melhor solução
        for i in range(tam_pop):
            if scores[i] < melhor_eval:
                melhor, melhor_eval = pop[i], scores[i] # o novo melhor e sua pontuação
                print(f"> Geração {gen}: Novo melhor {decodificado[i]} = {scores[i]}")
        
        scores_em_ordem = scores.copy()
        scores_em_ordem = sorted(scores_em_ordem)
        # Colocando a média dos 30 melhores na lista de bests
        bests.append(sum(scores_em_ordem[:30])/30)
    
    return [melhor, melhor_eval, bests]

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
melhor, score = alg_genetico(rosenbrock, limites, n_bits, geracoes, tam_pop, tx_cross, tx_mut, n_filhos)
print('Pronto!')
decodificado = decodifica(limites, n_bits, melhor)
print(f'({decodificado}) = {score}')