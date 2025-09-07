import os
import cv2
import numpy as np

# Load training data
cats, dogs = [], []
for f in os.listdir('Training Data/Cat')[:10]:
    if f.endswith('.jpg'):
        img = cv2.imread(f'Training Data/Cat/{f}', 0)
        if img is not None:
            cats.append(cv2.resize(img, (8, 8)).flatten() / 255.0)

for f in os.listdir('Training Data/Dog')[:10]:
    if f.endswith('.jpg'):
        img = cv2.imread(f'Training Data/Dog/{f}', 0)
        if img is not None:
            dogs.append(cv2.resize(img, (8, 8)).flatten() / 255.0)

cats, dogs = np.array(cats), np.array(dogs)

# Calculate statistics
cat_mean, cat_std = np.mean(cats, axis=0), np.std(cats, axis=0) + 1e-6
dog_mean, dog_std = np.mean(dogs, axis=0), np.std(dogs, axis=0) + 1e-6

# Load test image
img = cv2.imread('TestData/0.jpg', 0)
test_img = cv2.resize(img, (8, 8)).flatten() / 255.0

# Bayesian classification
def gaussian_prob(x, mean, std):
    return np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))

# Calculate likelihoods
cat_likelihood = np.prod(gaussian_prob(test_img, cat_mean, cat_std))
dog_likelihood = np.prod(gaussian_prob(test_img, dog_mean, dog_std))

# Priors (equal)
prior_cat = prior_dog = 0.5

# Posteriors
posterior_cat = cat_likelihood * prior_cat
posterior_dog = dog_likelihood * prior_dog

# Normalize
total = posterior_cat + posterior_dog
prob_cat = posterior_cat / total
prob_dog = posterior_dog / total

print("Bayesian Classification for 0.jpg:")
print(f"Prior Cat: {prior_cat:.1f}")
print(f"Prior Dog: {prior_dog:.1f}")
print(f"Likelihood Cat: {cat_likelihood:.2e}")
print(f"Likelihood Dog: {dog_likelihood:.2e}")
print(f"Posterior Cat: {prob_cat:.3f}")
print(f"Posterior Dog: {prob_dog:.3f}")
print(f"Prediction: {'Cat' if prob_cat > prob_dog else 'Dog'}")
