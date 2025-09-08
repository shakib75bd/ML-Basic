import numpy as np

# Sample data: Wind -> Weather (0=Calm/Clear, 1=Strong/Rainy)
wind = np.array([0, 1, 1, 0, 0, 1, 0, 1])
weather = np.array([0, 1, 1, 0, 0, 1, 0, 1])

print("Data (wind, weather):", list(zip(wind, weather)))

# Count and calculate probabilities from data
total = len(weather)
rainy_count = np.sum(weather == 1)
strong_rainy = np.sum((wind == 1) & (weather == 1))

# Prior and likelihood from data
prior_rainy = rainy_count / total
likelihood_strong_rainy = strong_rainy / rainy_count

print(f"Prior P(Rainy): {prior_rainy:.1f}")
print(f"Likelihood P(Strong|Rainy): {likelihood_strong_rainy:.1f}")

# Test prediction: if wind = 1 (strong)
test_wind = 1
if likelihood_strong_rainy > 0.5:
    prediction = "Rainy"
else:
    prediction = "Clear"

print(f"Input wind={test_wind} -> Weather: {prediction}")
