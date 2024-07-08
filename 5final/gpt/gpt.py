import numpy as np
import random
import optuna

def rosenbrock(x):
    """Rosenbrock function for optimization"""
    return sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)

def initialize_population(num_particles, dim, bounds):
    """Initialize particle positions and velocities"""
    pos = np.random.uniform(bounds[0], bounds[1], (num_particles, dim))
    vel = np.zeros((num_particles, dim))
    return pos, vel

def update_velocity(vel, pos, pbest, gbest, w, c1, c2, comm_matrix):
    """Update particle velocities"""
    r1, r2 = np.random.rand(), np.random.rand()
    cognitive = c1 * r1 * (pbest - pos)
    social = c2 * r2 * (gbest - pos) @ comm_matrix
    new_vel = w * vel + cognitive + social
    return new_vel

def apply_bounds(pos, bounds):
    """Ensure particles stay within bounds"""
    return np.clip(pos, bounds[0], bounds[1])

def hill_climb(func, pos, fitness, bounds, step_size=0.01, max_iter=100):
    """Simple hill climbing to refine global best"""
    improved = False
    dim = len(pos)
    for _ in range(max_iter):
        candidate = pos + step_size * np.random.randn(dim)
        candidate = apply_bounds(candidate, bounds)
        candidate_fitness = func(candidate)
        if candidate_fitness < fitness:
            pos, fitness = candidate, candidate_fitness
            improved = True
    return pos, fitness, improved

def cdeepso(func, num_particles=30, dim=10, bounds=[-5, 5], max_iter=100, w=0.5, c1=1.5, c2=1.5, t_mut=0.01, t_com=0.5, hill_climb_enabled=False):
    # Initialize particles
    pos, vel = initialize_population(num_particles, dim, bounds)
    pbest = np.copy(pos)
    pbest_fitness = np.apply_along_axis(func, 1, pbest)
    gbest = pbest[np.argmin(pbest_fitness)]
    gbest_fitness = np.min(pbest_fitness)
    comm_matrix = np.eye(dim) * (np.random.rand(dim, dim) < t_com)

    # Main loop
    for iter in range(max_iter):
        for i in range(num_particles):
            # Update velocity and position
            vel[i] = update_velocity(vel[i], pos[i], pbest[i], gbest, w, c1, c2, comm_matrix)
            pos[i] += vel[i]
            pos[i] = apply_bounds(pos[i], bounds)
            
            # Evaluate new fitness
            fitness = func(pos[i])
            if fitness < pbest_fitness[i]:
                pbest[i] = pos[i]
                pbest_fitness[i] = fitness
                if fitness < gbest_fitness:
                    gbest = pos[i]
                    gbest_fitness = fitness
        
        # Optionally apply hill climbing
        if hill_climb_enabled and iter >= 20 and iter < 30:
            hc_pos, hc_fitness, improved = hill_climb(func, gbest, gbest_fitness, bounds)
            if improved:
                gbest, gbest_fitness = hc_pos, hc_fitness
        
        yield gbest, gbest_fitness

def objective(trial):
    w = trial.suggest_uniform('w', 0.4, 0.9)
    c1 = trial.suggest_uniform('c1', 1.0, 2.5)
    c2 = trial.suggest_uniform('c2', 1.0, 2.5)
    t_mut = trial.suggest_uniform('t_mut', 0.01, 0.1)
    t_com = trial.suggest_uniform('t_com', 0.01, 0.5)
    num_particles = trial.suggest_int('num_particles', 10, 100)
    max_iter = trial.suggest_int('max_iter', 50, 500)
    
    best_fitness = float('inf')
    cdeepso_gen = cdeepso(rosenbrock, num_particles=num_particles, dim=10, bounds=[-5, 5], max_iter=max_iter, w=w, c1=c1, c2=c2, t_mut=t_mut, t_com=t_com)
    
    for gbest, gbest_fitness in cdeepso_gen:
        if gbest_fitness < best_fitness:
            best_fitness = gbest_fitness
    
    return best_fitness

# Run Optuna to find the best hyperparameters
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

print("Best hyperparameters: ", study.best_params)
print("Best score: ", study.best_value)
