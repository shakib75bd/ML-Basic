import os
import cv2
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score

# Load cat images
cats = []
for f in os.listdir('Training Data/Cat')[:15]:  # 15 images for training
    if f.endswith('.jpg'):
        img = cv2.imread(f'Training Data/Cat/{f}', 0)
        if img is not None:
            img = cv2.resize(img, (16, 16))  # 16x16 = 256 features
            cats.append(img.flatten() / 255.0)  # Normalize to 0-1

# Load dog images
dogs = []
for f in os.listdir('Training Data/Dog')[:15]:  # 15 images for training
    if f.endswith('.jpg'):
        img = cv2.imread(f'Training Data/Dog/{f}', 0)
        if img is not None:
            img = cv2.resize(img, (16, 16))  # 16x16 = 256 features
            dogs.append(img.flatten() / 255.0)  # Normalize to 0-1

# Create training data
X_train = np.array(cats + dogs)
y_train = np.array([0]*len(cats) + [1]*len(dogs))  # 0=cat, 1=dog

# Create and train LDA
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)

# === PREDICT TEST IMAGES ===

for filename in sorted(os.listdir('TestData')):
    if filename.endswith('.jpg'):
        # Load test image
        img = cv2.imread(f'TestData/{filename}', 0)
        if img is not None:
            img = cv2.resize(img, (16, 16))
            x_test = img.flatten() / 255.0

            # Predict and get probabilities
            prediction = lda.predict([x_test])[0]
            probabilities = lda.predict_proba([x_test])[0]

            # Convert to class label
            result = "Cat" if prediction == 0 else "Dog"
            cat_prob = probabilities[0]
            dog_prob = probabilities[1]
            decision = "Cat" if cat_prob > dog_prob else "Dog"

            print(f"{filename}: {result}")
            print(f"  Cat probability: {cat_prob:.3f}")
            print(f"  Dog probability: {dog_prob:.3f}")
            print(f"  Decision: {decision}")
            print()
