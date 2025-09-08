import numpy as np

print("Naive Bayes - Weather Prediction")
print("=" * 30)

# Data: [Pressure, Wind] -> Weather (0=Clear, 1=Rainy)
X = np.array([[1,0], [0,1], [1,1], [0,0], [1,0], [0,1], [1,1], [0,0]])
y = np.array([0, 1, 1, 0, 0, 1, 1, 0])  # 0=Clear, 1=Rainy

print("Training Data: [Pressure, Wind] -> Weather")
print("Pressure: 0=Low, 1=High | Wind: 0=Calm, 1=Strong")
print("Weather: 0=Clear, 1=Rainy")
for i in range(len(X)):
    pressure = "High" if X[i][0] == 1 else "Low"
    wind = "Strong" if X[i][1] == 1 else "Calm"
    weather = "Rainy" if y[i] == 1 else "Clear"
    print(f"{pressure} pressure, {wind} wind -> {weather}")

# Calculate probabilities
clear_data = X[y == 0]
rainy_data = X[y == 1]
prior_clear = np.mean(y == 0)
prior_rainy = np.mean(y == 1)

# Test case: High pressure, Strong wind [1,1]
high_pressure_clear = np.mean(clear_data[:, 0] == 1)
high_pressure_rainy = np.mean(rainy_data[:, 0] == 1)
strong_wind_clear = np.mean(clear_data[:, 1] == 1)
strong_wind_rainy = np.mean(rainy_data[:, 1] == 1)

print(f"\nPriors: P(Clear)={prior_clear:.1f}, P(Rainy)={prior_rainy:.1f}")
print(f"P(High pressure|Clear)={high_pressure_clear:.1f}")
print(f"P(High pressure|Rainy)={high_pressure_rainy:.1f}")
print(f"P(Strong wind|Clear)={strong_wind_clear:.1f}")
print(f"P(Strong wind|Rainy)={strong_wind_rainy:.1f}")

# Predict weather for [1,1] (High pressure, Strong wind)
prob_clear = prior_clear * high_pressure_clear * strong_wind_clear
prob_rainy = prior_rainy * high_pressure_rainy * strong_wind_rainy
total = prob_clear + prob_rainy

print(f"\nTest [1,1]: High pressure, Strong wind")
print(f"P(Clear) = {prob_clear/total:.3f}")
print(f"P(Rainy) = {prob_rainy/total:.3f}")
print(f"Weather prediction: {'Rainy' if prob_rainy > prob_clear else 'Clear'}")
