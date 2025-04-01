#PSO
import numpy as np

def pso(f, n_particles=20, dim=2, max_iter=100, bounds=(-5, 5)):
    particles = np.random.uniform(bounds[0], bounds[1], (n_particles, dim))
    velocities = np.zeros((n_particles, dim))
    p_best = particles.copy()
    g_best = particles[np.argmin([f(p) for p in particles])]

    w, c1, c2 = 0.7, 1.5, 1.5

    for _ in range(max_iter):
        r1, r2 = np.random.rand(2)
        velocities = (w * velocities +
                     c1 * r1 * (p_best - particles) +
                     c2 * r2 * (g_best - particles))
        particles += velocities
        particles = np.clip(particles, bounds[0], bounds[1])

        fitness = np.array([f(p) for p in particles])
        p_best[fitness < [f(p) for p in p_best]] = particles[fitness < [f(p) for p in p_best]]
        g_best = particles[np.argmin(fitness)]

    return g_best, f(g_best)

# Example usage
f = lambda x: sum(x**2)  # Simple sphere function
best_pos, best_val = pso(f)
print(f"Best position: {best_pos}, Best value: {best_val}")
