import numpy as np

# Sample data: Wind -> Weather (0=Calm/Clear, 1=Strong/Rainy)
wind = np.array([0, 1, 1, 0, 0, 1, 0, 1])
weather = np.array([0, 1, 1, 0, 0, 1, 0, 1])

print("Data (wind, weather):", list(zip(wind, weather)))

# Count and calculate probabilities from data
total = len(weather)
rainy_count = np.sum(weather == 1)
clear_count = np.sum(weather == 0)
strong_rainy = np.sum((wind == 1) & (weather == 1))
strong_clear = np.sum((wind == 1) & (weather == 0))
calm_rainy = np.sum((wind == 0) & (weather == 1))
calm_clear = np.sum((wind == 0) & (weather == 0))

# Calculate all probabilities
prior_rainy = rainy_count / total
prior_clear = clear_count / total
likelihood_strong_rainy = strong_rainy / rainy_count if rainy_count > 0 else 0
likelihood_strong_clear = strong_clear / clear_count if clear_count > 0 else 0
likelihood_calm_rainy = calm_rainy / rainy_count if rainy_count > 0 else 0
likelihood_calm_clear = calm_clear / clear_count if clear_count > 0 else 0

print(f"\nPriors:")
print(f"P(Rainy): {prior_rainy:.1f}")
print(f"P(Clear): {prior_clear:.1f}")

print(f"\nLikelihoods:")
print(f"P(Strong|Rainy): {likelihood_strong_rainy:.1f}")
print(f"P(Strong|Clear): {likelihood_strong_clear:.1f}")
print(f"P(Calm|Rainy): {likelihood_calm_rainy:.1f}")
print(f"P(Calm|Clear): {likelihood_calm_clear:.1f}")

# Test prediction - Change this value to test different scenarios
test_wind = 1  # 0=Calm, 1=Strong

if test_wind == 1:  # Strong wind
    # Calculate posterior probabilities
    posterior_rainy = likelihood_strong_rainy * prior_rainy
    posterior_clear = likelihood_strong_clear * prior_clear
    wind_type = "Strong"
else:  # Calm wind
    posterior_rainy = likelihood_calm_rainy * prior_rainy
    posterior_clear = likelihood_calm_clear * prior_clear
    wind_type = "Calm"

# Normalize posteriors
total_posterior = posterior_rainy + posterior_clear
if total_posterior > 0:
    prob_rainy = posterior_rainy / total_posterior
    prob_clear = posterior_clear / total_posterior
else:
    prob_rainy = prob_clear = 0.5

prediction = "Rainy" if prob_rainy > prob_clear else "Clear"

print(f"\nInput: {wind_type} wind ({test_wind})")
print(f"Posterior P(Rainy|{wind_type}): {prob_rainy:.2f}")
print(f"Posterior P(Clear|{wind_type}): {prob_clear:.2f}")
print(f"Prediction: {prediction}")
