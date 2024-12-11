import numpy as np
import matplotlib.pyplot as plt

# Genetic Algorithm Parameters
population_size = 20
num_generations = 500
mutation_rate = 0.2
image_resolution = (250, 250)

# Fitness Function: Combine Global Contrast Factor (GCF) and Information Theory (IT)
def fitness_function(image):
    gcf = global_contrast_factor(image)
    it = information_theory(image)
    return gcf + it

# Example Global Contrast Factor (GCF)
def global_contrast_factor(image):
    luminance = np.mean(image, axis=2)  # Average across RGB channels
    contrast = np.std(luminance)  # Standard deviation as a simple contrast measure
    return contrast

# Example Information Theory (IT)
def information_theory(image):
    histogram, _ = np.histogram(image.flatten(), bins=256, range=(0, 255), density=True)
    entropy = -np.sum(histogram * np.log2(histogram + 1e-10))  # Shannon entropy
    return entropy

# Initialize Population
def initialize_population(size, resolution):
    return [np.random.randint(0, 256, (*resolution, 3), dtype=np.uint8) for _ in range(size)]

# Crossover Operation
def crossover(parent1, parent2):
    crossover_point = np.random.randint(0, parent1.shape[1])
    child = np.concatenate((parent1[:, :crossover_point], parent2[:, crossover_point:]), axis=1)
    return child

# Mutation Operation
def mutate(image, rate):
    mutated_image = image.copy()
    num_pixels = np.prod(image.shape[:2])
    num_mutations = int(rate * num_pixels)
    for _ in range(num_mutations):
        x, y = np.random.randint(0, image.shape[0]), np.random.randint(0, image.shape[1])
        mutated_image[x, y] = np.random.randint(0, 256, 3)
    return mutated_image

# Main GA Loop
population = initialize_population(population_size, image_resolution)
for generation in range(num_generations):
    print(f"Generation {generation + 1}: Evaluating fitness")

    # Evaluate Fitness
    fitness_scores = [fitness_function(image) for image in population]
    if len(fitness_scores) != len(population):
        raise ValueError("Mismatch between fitness scores and population size")
    best_fitness = max(fitness_scores)
    print(f"  Best fitness: {best_fitness:.2f}")

    # Select Parents (Roulette Wheel Selection)
    total_fitness = sum(fitness_scores)
    probabilities = [score / total_fitness for score in fitness_scores]
    indices = np.arange(len(population))  # Indices for the population
    selected_indices = np.random.choice(indices, size=population_size, p=probabilities, replace=True)
    parents = [population[i] for i in selected_indices]

    print("  Parents selected")

    # Generate New Population
    new_population = []
    for i in range(0, len(parents), 2):
        parent1, parent2 = parents[i], parents[(i + 1) % len(parents)]
        child = crossover(parent1, parent2)
        child = mutate(child, mutation_rate)
        new_population.append(child)
    population = new_population

    print("  New population generated")

    # Safely find the best image
    best_index = np.argmax(fitness_scores)
    if best_index >= len(population):
        raise ValueError("Best index is out of range")
    best_image = population[best_index]

    # Display the best image
    plt.imshow(best_image)
    plt.title(f"Generation {generation + 1} - Best Fitness: {best_fitness:.2f}")
    plt.axis('off')
    plt.show()

print("Evolutionary art generation complete!")
