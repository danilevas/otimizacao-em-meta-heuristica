import numpy as np

# Rosenbrock function
def rosenbrock(x):
    return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

# Fitness function
def fitness(individual):
    return 1 / (1 + rosenbrock(individual))

# Initialize population
def initialize_population(population_size, bounds):
    population = []
    for _ in range(population_size):
        individual = np.random.uniform(bounds[0], bounds[1], size=(2,))
        population.append(individual)
    return population

# Select parents using roulette wheel selection
def roulette_wheel_selection(population, fitness_values):
    total_fitness = sum(fitness_values)
    probabilities = [fitness_val / total_fitness for fitness_val in fitness_values]
    selected_parents_indices = np.random.choice(len(population), size=2, p=probabilities, replace=False)
    selected_parents = [population[i] for i in selected_parents_indices]
    return selected_parents

# Crossover
def crossover(parent1, parent2):
    alpha = np.random.rand()
    child1 = alpha * parent1 + (1 - alpha) * parent2
    child2 = (1 - alpha) * parent1 + alpha * parent2
    return child1, child2

# Mutation
def mutation(individual, mutation_rate, bounds):
    mutated_individual = np.copy(individual)
    for i in range(len(individual)):
        if np.random.rand() < mutation_rate:
            mutated_individual[i] += np.random.uniform(-0.1, 0.1) # Mutation amount
            mutated_individual[i] = np.clip(mutated_individual[i], bounds[0], bounds[1])
    return mutated_individual

# Genetic algorithm
def genetico_gpt(population_size, bounds, generations, mutation_rate):
    population = initialize_population(population_size, bounds)
    for _ in range(generations):
        fitness_values = [fitness(individual) for individual in population]

        new_population = []
        for _ in range(population_size // 2):
            parent1, parent2 = roulette_wheel_selection(population, fitness_values)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutation(child1, mutation_rate, bounds)
            child2 = mutation(child2, mutation_rate, bounds)
            new_population.extend([child1, child2])
        
        population = new_population
    
    best_individual = min(population, key=lambda x: rosenbrock(x))
    return best_individual

# Parameters
population_size = 100
bounds = (-2.048, 2.048)
generations = 100
mutation_rate = 0.1

# Run genetic algorithm
best_solution = genetico_gpt(population_size, bounds, generations, mutation_rate)
print("Best solution:", best_solution)
print("Fitness:", fitness(best_solution))
print("Objective value:", rosenbrock(best_solution))
