import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load cat images
cats = []
for f in os.listdir('Training Data/Cat')[:15]:  # 15 images for better training
    if f.endswith('.jpg'):  # Only process jpg files
        img = cv2.imread(f'Training Data/Cat/{f}', 0)
        if img is not None:  # Check if image loaded successfully
            img = cv2.resize(img, (16, 16))  # Larger size: 16x16 = 256 pixels
            cats.append(img.flatten() / 255.0)  # Normalize to 0-1

# Load dog images
dogs = []
for f in os.listdir('Training Data/Dog')[:15]:  # 15 images for better training
    if f.endswith('.jpg'):  # Only process jpg files
        img = cv2.imread(f'Training Data/Dog/{f}', 0)
        if img is not None:  # Check if image loaded successfully
            img = cv2.resize(img, (16, 16))  # Larger size: 16x16 = 256 pixels
            dogs.append(img.flatten() / 255.0)  # Normalize to 0-1

# Create training data
X_train = np.array(cats + dogs)
y_train = np.array([0]*len(cats) + [1]*len(dogs))  # 0=cat, 1=dog

print(f"Training data: {X_train.shape}")
print(f"Cats: {len(cats)}, Dogs: {len(dogs)}")

# === SIMPLE SVM USING SCIKIT-LEARN ===

# Create and train SVM
svm = SVC(kernel='rbf', random_state=42)
svm.fit(X_train, y_train)

# Check training accuracy
train_predictions = svm.predict(X_train)
accuracy = accuracy_score(y_train, train_predictions)
print(f"Training accuracy: {accuracy:.2f}")

# === PREDICT TEST IMAGES ===

print("\nPredictions for ALL test images:")
print("-" * 50)
for filename in sorted(os.listdir('TestData')):
    if filename.endswith('.jpg'):
        # Load test image
        img = cv2.imread(f'TestData/{filename}', 0)
        if img is not None:  # Check if image loaded successfully
            img = cv2.resize(img, (16, 16))
            x_test = img.flatten() / 255.0

            # Predict using SVM
            prediction = svm.predict([x_test])[0]
            confidence = svm.decision_function([x_test])[0]

            # Convert to class label
            result = "Cat" if prediction == 0 else "Dog"
            print(f"{filename}: {result} (confidence: {abs(confidence):.2f})")

print(f"\nSVM Summary:")
print(f"Training samples: {len(X_train)}")
print(f"Features per image: {X_train.shape[1]}")
print(f"Support vectors: {len(svm.support_)}")
print(f"Training accuracy: {accuracy:.2f}")
