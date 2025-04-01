#Fuzzy Logic
import numpy as np

def fuzzy_membership(x, points):
    """Simple triangular membership function"""
    a, b, c = points
    if x <= a or x >= c:
        return 0
    if a < x <= b:
        return (x - a) / (b - a)
    if b < x < c:
        return (c - x) / (c - b)
    return 0

def fuzzy_system(service_quality):
    # Define membership functions
    poor = lambda x: fuzzy_membership(x, [0, 0, 5])
    good = lambda x: fuzzy_membership(x, [0, 5, 10])
    excellent = lambda x: fuzzy_membership(x, [5, 10, 10])

    low_tip = lambda x: fuzzy_membership(x, [0, 0, 10])
    medium_tip = lambda x: fuzzy_membership(x, [0, 10, 20])
    high_tip = lambda x: fuzzy_membership(x, [10, 20, 20])

    # Fuzzification
    poor_val = poor(service_quality)
    good_val = good(service_quality)
    excellent_val = excellent(service_quality)

    # Rules (min-max inference)
    # Rule 1: If poor service -> low tip
    # Rule 2: If good service -> medium tip
    # Rule 3: If excellent service -> high tip
    low_rule = poor_val
    medium_rule = good_val
    high_rule = excellent_val

    # Defuzzification (Centroid method)
    x = np.linspace(0, 20, 100)  # Tip range
    low_mem = np.minimum(low_rule, [low_tip(xi) for xi in x])
    medium_mem = np.minimum(medium_rule, [medium_tip(xi) for xi in x])
    high_mem = np.minimum(high_rule, [high_tip(xi) for xi in x])

    # Aggregate output
    aggregated = np.maximum.reduce([low_mem, medium_mem, high_mem])

    # Centroid calculation
    if np.sum(aggregated) == 0:  # Avoid division by zero
        return 0
    centroid = np.sum(x * aggregated) / np.sum(aggregated)

    return centroid

# Example usage
if __name__ == "__main__":
    service = 7.5  # Service quality (0-10)
    tip = fuzzy_system(service)
    print(f"Service quality: {service}")
    print(f"Suggested tip: {tip:.2f}%")
