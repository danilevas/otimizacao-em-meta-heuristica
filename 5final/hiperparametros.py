import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import time
from final import *
from analise import *
import optuna

# Define the objective function for Optuna
def objective(trial):
    w_i = trial.suggest_float('w_i', 0.4, 0.9)
    w_m = trial.suggest_float('w_m', 0.4, 0.9)
    w_s = trial.suggest_float('w_s', 0.4, 0.9)
    t_mut = trial.suggest_float('t_mut', 0.01, 0.1)
    t_com = trial.suggest_float('t_com', 0.01, 0.5)
    num_particulas = trial.suggest_int('num_particulas', 10, 100)
    # max_iter = trial.suggest_int('max_iter', 50, 500)

    best_fitness = float('inf')
    cdeepso_gen = meu_cdeepso(rosenbrock_n, w_i, w_m, w_s, t_mut, t_com, com_hill_climb=True, dim=10, num_particulas=num_particulas, max_iter=100)
    for x_gb, x_gb_fitness, _ in cdeepso_gen:
        if x_gb_fitness < best_fitness:
            best_fitness = x_gb_fitness
    
    return best_fitness

# Run Optuna to find the best hyperparameters
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

print("Best hyperparameters: ", study.best_params)
print("Best score: ", study.best_value)