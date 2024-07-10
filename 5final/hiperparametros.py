import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import time
from final import *
from analise import *
import optuna

def ajeita_params(dict):
    texto = ""
    for key, value in dict.items():
        texto += f"{key}={value}, "
    return texto

# Define the objective function for Optuna
def objective(trial, algoritmo, funcao, dim):
    w_i = trial.suggest_float('w_i', 0.01, 0.9)
    w_m = trial.suggest_float('w_m', 0.01, 0.9)
    w_s = trial.suggest_float('w_s', 0.01, 0.9)
    t_mut = trial.suggest_float('t_mut', 0.01, 1)
    t_com = trial.suggest_float('t_com', 0.01, 1)
    if algoritmo == meu_cdeepso: F = trial.suggest_float('F', 0.01, 2)
    num_particulas = trial.suggest_int('num_particulas', 10, 100)
    # max_iter = trial.suggest_int('max_iter', 50, 500)

    best_fitness = float('inf')
    if algoritmo == meu_cdeepso: cdeepso_gen = meu_cdeepso(funcao, w_i, w_m, w_s, t_mut, t_com, F, com_hill_climb=True,
                                                           dim=dim, num_particulas=num_particulas, max_iter=100, opcao=1)
    else: cdeepso_gen = meu_deepso(funcao, w_i, w_m, w_s, t_mut, t_com, com_hill_climb=True, dim=dim, num_particulas=num_particulas, max_iter=100)
    
    for x_gb, x_gb_fitness, _ in cdeepso_gen:
        if x_gb_fitness < best_fitness:
            best_fitness = x_gb_fitness
    
    return best_fitness

# Run Optuna to find the best hyperparameters
study = optuna.create_study(direction='minimize')
study.optimize(lambda trial: objective(trial, meu_cdeepso, rastrigin_n, dim=50), n_trials=50)

print("Best hyperparameters: ", ajeita_params(study.best_params))
print("Best score: ", study.best_value)