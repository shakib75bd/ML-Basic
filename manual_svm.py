import os
import cv2
import numpy as np

class SimplestSVM:
    def __init__(self):
        self.w = None  # Weight vector (hyperplane normal)
        self.b = None  # Bias term (hyperplane offset)

    def fit(self, X, y):
        """Train SVM using basic matrix operations"""
        n_samples, n_features = X.shape

        # Convert labels: 0,1 -> -1,1 (SVM standard)
        y = np.where(y == 0, -1, 1)

        # Initialize weight vector and bias
        self.w = np.zeros(n_features)
        self.b = 0

        # Training parameters
        learning_rate = 0.01
        epochs = 100

        print("Training SVM...")

        # Training loop
        for epoch in range(epochs):
            for i in range(n_samples):
                # Calculate decision: w*x + b
                decision = np.dot(X[i], self.w) + self.b

                # Check if point is misclassified or within margin
                if y[i] * decision < 1:
                    # Update weights: w = w + learning_rate * y[i] * x[i]
                    self.w = self.w + learning_rate * y[i] * X[i]
                    # Update bias: b = b + learning_rate * y[i]
                    self.b = self.b + learning_rate * y[i]

            if epoch % 20 == 0:
                print(f"Epoch {epoch}/100")

    def predict(self, X):
        """Predict using: sign(w*x + b)"""
        # Matrix multiplication: decision = X * w + b
        decision = np.dot(X, self.w) + self.b
        # Convert back to 0,1 labels
        return np.where(decision >= 0, 1, 0)

def load_images(folder, max_images=20):
    """Load and flatten images"""
    images = []
    files = [f for f in os.listdir(folder) if f.endswith('.jpg')]

    for i, filename in enumerate(files[:max_images]):
        img_path = os.path.join(folder, filename)
        # Load grayscale image
        img = cv2.imread(img_path, 0)
        # Resize to small size
        img = cv2.resize(img, (16, 16))
        # Flatten to 1D array
        images.append(img.flatten())

    return np.array(images)

# === MAIN PROGRAM ===

# Load training data
print("Loading training data...")
cats = load_images('Training Data/Cat', 20)
dogs = load_images('Training Data/Dog', 20)

# Create training matrix X and labels y
X = np.vstack([cats, dogs])        # Stack cat and dog images
y = np.hstack([np.zeros(len(cats)), np.ones(len(dogs))])  # 0=cat, 1=dog

print(f"Training data: {X.shape[0]} images, {X.shape[1]} features each")

# Normalize data (important for SVM)
X = X / 255.0  # Scale pixels to 0-1 range

# Train SVM
svm = SimplestSVM()
svm.fit(X, y)

# Load test data
print("\nLoading test data...")
test_images = []
test_names = []
for filename in sorted(os.listdir('TestData')):
    if filename.endswith('.jpg'):
        img_path = os.path.join('TestData', filename)
        img = cv2.imread(img_path, 0)
        img = cv2.resize(img, (16, 16))
        test_images.append(img.flatten())
        test_names.append(filename)

X_test = np.array(test_images) / 255.0  # Same normalization

# Make predictions
predictions = svm.predict(X_test)

# Show results
print("\nPredictions:")
print("-" * 30)
for name, pred in zip(test_names, predictions):
    label = "Cat" if pred == 0 else "Dog"
    print(f"{name}: {label}")

print(f"\nSVM learned:")
print(f"Weight vector size: {len(svm.w)}")
print(f"Bias: {svm.b:.3f}")
