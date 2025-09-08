import numpy as np

# Sample data: Hours studied vs Test score
X = np.array([1, 2, 3, 4, 5, 6, 7, 8])
y = np.array([2, 4, 6, 8, 10, 12, 14, 16])

print("Least Squares Linear Regression")
print("Data:", list(zip(X, y)))

# Calculate slope (m) and intercept (b)
n = len(X)
sum_x, sum_y, sum_xy, sum_x2 = np.sum(X), np.sum(y), np.sum(X * y), np.sum(X * X)
m = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
b = (sum_y - m * sum_x) / n

print(f"Equation: y = {m:.1f}x + {b:.1f}")

# Predictions and accuracy
y_pred = m * X + b
mse = np.mean((y - y_pred) ** 2)
r_squared = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)

print(f"MSE: {mse:.6f}, R²: {r_squared:.6f} ({r_squared*100:.1f}% accuracy)")

# Test prediction and sample results
test_pred = m * 5.5 + b
print(f"Prediction for 5.5 hours: {test_pred:.1f} points")
print("Sample predictions:")
for i in [0, 3, 7]:
    print(f"  {X[i]} hours → actual: {y[i]}, predicted: {y_pred[i]:.1f}")
print(f"Rate: {m:.1f} points per hour")
