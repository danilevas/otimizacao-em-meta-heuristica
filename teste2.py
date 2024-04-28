from numpy.random import randint
from numpy.random import rand
import numpy.random as npr

def roleta_dupla(pop, scores):
    scores_inv = [1/s for s in scores]
    max = sum(scores_inv)
    probs_selecao = [c/max for c in scores_inv]
    pai1 = pop[npr.choice(len(pop), p=probs_selecao)]
    pai2 = pop[npr.choice(len(pop), p=probs_selecao)]
    while pai1 == pai2:
        pai2 = pop[npr.choice(len(pop), p=probs_selecao)]
    return pai1, pai2

pop = ["banana", "maçã", "batata", "laranja", "cHucHsu", "(((<<<000>>>)))"]
scores = [10, 20, 30, 38, 1.5, 0.5]
for i in range(100):
    print(roleta_dupla(pop, scores))