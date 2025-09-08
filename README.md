# ML-Basic: Simple Machine Learning Implementations

Six simplified machine learning algorithms with clean implementations for educational purposes.

## What This Project Has

**6 Machine Learning Algorithms:**

| Algorithm     | Type                     | Description                            |
| ------------- | ------------------------ | -------------------------------------- |
| SVM           | Classification           | Support Vector Machine with RBF kernel |
| Bayesian      | Classification           | Naive Bayes with probability estimates |
| LDA           | Classification           | Linear Discriminant Analysis           |
| Least Squares | Classification           | Linear regression for classification   |
| PCA           | Dimensionality Reduction | Principal Component Analysis           |
| SVD           | Matrix Factorization     | Singular Value Decomposition           |

**Data:**

- Cat/Dog training images (for classification algorithms)
- Test images (0.jpg to 9.jpg)
- Sample matrices (for PCA and SVD)

## How to Run

**Setup with Virtual Environment:**

```bash
git clone <repository-url>
cd MLLab

# Create virtual environment
python -m venv ml_env

# Activate virtual environment
# On macOS/Linux:
source ml_env/bin/activate
# On Windows:
ml_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Run Algorithms:**

```bash
python svm.py        # Support Vector Machine
python bayesian.py   # Bayesian Classification
python lda.py        # Linear Discriminant Analysis
python least_square.py # Least Squares
python pca.py        # Principal Component Analysis
python svd.py        # Singular Value Decomposition
```

**Requirements:**

- Python 3.7+
- Dependencies: numpy, opencv-python, scikit-learn

Each script runs independently and produces immediate results.
