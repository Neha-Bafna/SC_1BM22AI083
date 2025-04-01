#Genetic Algorithm
import numpy as np

def genetic_algorithm(pop_size=100, chromosome_length=10, generations=50, mutation_rate=0.1):
    # Ensure pop_size is even for simpler crossover
    pop_size = pop_size if pop_size % 2 == 0 else pop_size + 1

    # Initialize random population (0s and 1s)
    population = np.random.randint(0, 2, (pop_size, chromosome_length))

    for _ in range(generations):
        # Fitness: convert binary to decimal
        fitness = np.array([int(''.join(map(str, ind)), 2) for ind in population])

        # Selection: tournament selection
        parents = []
        for _ in range(pop_size // 2):
            tournament = np.random.choice(pop_size, 2)
            winner = tournament[np.argmax(fitness[tournament])]
            parents.append(population[winner])
        parents = np.array(parents)

        # Crossover
        offspring = []
        for i in range(0, len(parents), 2):
            if i + 1 < len(parents):  # Ensure we have pairs
                cut = np.random.randint(1, chromosome_length)
                child1 = np.concatenate([parents[i][:cut], parents[i+1][cut:]])
                child2 = np.concatenate([parents[i+1][:cut], parents[i][cut:]])
                offspring.extend([child1, child2])
        offspring = np.array(offspring)

        # Mutation
        mutation_mask = np.random.random(offspring.shape) < mutation_rate
        offspring ^= mutation_mask  # Flip bits where mask is True

        # Elitism: keep best individual and fill with offspring
        best_idx = np.argmax(fitness)
        new_population = [population[best_idx]]
        new_population.extend(offspring[:pop_size-1])  # Take only what fits
        population = np.array(new_population)

        # Ensure population size stays consistent
        if len(population) < pop_size:
            extra = np.random.randint(0, 2, (pop_size - len(population), chromosome_length))
            population = np.vstack([population, extra])

    # Final evaluation
    fitness = np.array([int(''.join(map(str, ind)), 2) for ind in population])
    best_idx = np.argmax(fitness)

    return population[best_idx], fitness[best_idx]

# Run the algorithm
if __name__ == "__main__":
    try:
        best_chromosome, best_fitness = genetic_algorithm()
        print(f"Best chromosome: {list(best_chromosome)}")
        print(f"Best fitness (decimal value): {best_fitness}")
        print(f"Expected maximum possible: {2**10 - 1}")  # 1023 for length 10
    except Exception as e:
        print(f"An error occurred: {e}")
