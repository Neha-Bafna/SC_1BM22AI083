#2. Grey wolf
import numpy as np

def gwo(f, n_wolves=10, dim=2, max_iter=100, bounds=(-5, 5)):
    # Initialize wolf pack randomly within bounds
    wolves = np.random.uniform(bounds[0], bounds[1], (n_wolves, dim))

    # Initialize alpha, beta, delta (best three wolves)
    fitness = np.array([f(w) for w in wolves])
    alpha_idx = np.argmin(fitness)
    alpha = wolves[alpha_idx].copy()
    alpha_score = fitness[alpha_idx]

    beta_idx = np.argsort(fitness)[1]  # Second best
    beta = wolves[beta_idx].copy()

    delta_idx = np.argsort(fitness)[2]  # Third best
    delta = wolves[delta_idx].copy()

    # Main optimization loop
    for t in range(max_iter):
        a = 2 - 2 * t / max_iter  # Linearly decrease a from 2 to 0

        for i in range(n_wolves):
            # Random coefficients
            r1, r2 = np.random.rand(2)
            A1 = 2 * a * r1 - a  # Coefficient for alpha
            C1 = 2 * r2          # Coefficient for alpha

            r1, r2 = np.random.rand(2)
            A2 = 2 * a * r1 - a  # Coefficient for beta
            C2 = 2 * r2          # Coefficient for beta

            r1, r2 = np.random.rand(2)
            A3 = 2 * a * r1 - a  # Coefficient for delta
            C3 = 2 * r2          # Coefficient for delta

            # Update wolf position based on alpha, beta, delta
            D_alpha = abs(C1 * alpha - wolves[i])
            D_beta = abs(C2 * beta - wolves[i])
            D_delta = abs(C3 * delta - wolves[i])

            X1 = alpha - A1 * D_alpha
            X2 = beta - A2 * D_beta
            X3 = delta - A3 * D_delta

            wolves[i] = (X1 + X2 + X3) / 3  # Average position

            # Clip to bounds
            wolves[i] = np.clip(wolves[i], bounds[0], bounds[1])

        # Update fitness and leaders
        fitness = np.array([f(w) for w in wolves])
        sorted_idx = np.argsort(fitness)

        if fitness[sorted_idx[0]] < alpha_score:
            alpha_score = fitness[sorted_idx[0]]
            alpha = wolves[sorted_idx[0]].copy()

        beta = wolves[sorted_idx[1]].copy()
        delta = wolves[sorted_idx[2]].copy()

    return alpha, alpha_score

# Example usage
if __name__ == "__main__":
    # Objective function: sphere function f(x) = sum(x^2)
    f = lambda x: np.sum(x**2)

    # Run GWO
    best_position, best_value = gwo(f, n_wolves=10, dim=2, max_iter=50)
    print(f"Best position: {best_position}")
    print(f"Best value: {best_value}")
