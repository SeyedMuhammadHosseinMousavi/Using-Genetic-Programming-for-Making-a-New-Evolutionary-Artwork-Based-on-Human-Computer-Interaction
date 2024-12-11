import numpy as np
import matplotlib.pyplot as plt

# Particle Swarm Optimization (PSO) Parameters
population_size = 30
num_iterations = 100
inertia = 0.5
c1 = 2.0  # Cognitive parameter
c2 = 2.0  # Social parameter
image_resolution = (300, 300)

# Fitness Function: Enhanced to reward spiral-like patterns and symmetry
def fitness_function(image):
    complexity = calculate_complexity(image)
    symmetry = calculate_symmetry(image)
    contrast = global_contrast_factor(image)
    return complexity + symmetry + contrast

# Calculate Complexity using edge detection
def calculate_complexity(image):
    gradient_x = np.abs(np.diff(image, axis=0))
    gradient_y = np.abs(np.diff(image, axis=1))
    complexity = np.sum(gradient_x) + np.sum(gradient_y)
    return complexity / image.size

# Global Contrast Factor (GCF)
def global_contrast_factor(image):
    luminance = np.mean(image, axis=2)  # Average across RGB channels
    contrast = np.std(luminance)  # Standard deviation as a simple contrast measure
    return contrast

# Calculate Symmetry
def calculate_symmetry(image):
    vertical_symmetry = np.sum(np.abs(image - np.flip(image, axis=1)))
    horizontal_symmetry = np.sum(np.abs(image - np.flip(image, axis=0)))
    total_symmetry = -(vertical_symmetry + horizontal_symmetry) / image.size  # Negate to reward symmetry
    return total_symmetry

# Initialize Population with spiral-like patterns
def initialize_population(size, resolution):
    population = []
    for _ in range(size):
        x = np.linspace(-1.0, 1.0, resolution[0])
        y = np.linspace(-1.0, 1.0, resolution[1])
        x, y = np.meshgrid(x, y)
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        spiral = (np.sin(10 * r + 5 * theta) * 127 + 128).astype(np.uint8)
        image = np.stack([spiral, spiral, spiral], axis=2)  # Grayscale to RGB
        population.append(image)
    return population

# PSO Update Function
def update_particles(positions, velocities, personal_best_positions, global_best_position, inertia, c1, c2):
    for i in range(len(positions)):
        r1, r2 = np.random.random(), np.random.random()
        cognitive_component = c1 * r1 * (personal_best_positions[i] - positions[i])
        social_component = c2 * r2 * (global_best_position - positions[i])
        velocities[i] = inertia * velocities[i] + cognitive_component + social_component
        positions[i] = np.clip(positions[i] + velocities[i], 0, 255)  # Keep positions within valid range

# Main PSO Loop
population = initialize_population(population_size, image_resolution)
velocities = [np.random.uniform(-1, 1, (image_resolution[0], image_resolution[1], 3)) for _ in range(population_size)]
personal_best_positions = population[:]
personal_best_scores = [fitness_function(image) for image in population]
global_best_position = personal_best_positions[np.argmax(personal_best_scores)]
global_best_score = max(personal_best_scores)

for iteration in range(num_iterations):
    print(f"Iteration {iteration + 1}: Evaluating fitness")

    # Evaluate Fitness
    fitness_scores = [fitness_function(image) for image in population]

    for i in range(population_size):
        if fitness_scores[i] > personal_best_scores[i]:
            personal_best_scores[i] = fitness_scores[i]
            personal_best_positions[i] = population[i]

    best_particle_index = np.argmax(personal_best_scores)
    if personal_best_scores[best_particle_index] > global_best_score:
        global_best_score = personal_best_scores[best_particle_index]
        global_best_position = personal_best_positions[best_particle_index]

    print(f"  Best fitness: {global_best_score:.2f}")

    # Update Particles
    update_particles(population, velocities, personal_best_positions, global_best_position, inertia, c1, c2)

    # Display the best image
    plt.imshow(global_best_position.astype(np.uint8), cmap='inferno')
    plt.title(f"Iteration {iteration + 1} - Best Fitness: {global_best_score:.2f}")
    plt.axis('off')
    plt.pause(0.5)

print("Evolutionary art generation with PSO complete!")