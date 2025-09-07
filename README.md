# ML-Basic: Simple Machine Learning Implementations

Six simplified machine learning algorithms with clean implementations for educational purposes.

## What This Project Has

**6 Machine Learning Algorithms:**

| Algorithm     | Type                     | Accuracy | Description                            |
| ------------- | ------------------------ | -------- | -------------------------------------- |
| SVM           | Classification           | 100%     | Support Vector Machine with RBF kernel |
| Bayesian      | Classification           | 92.9%    | Naive Bayes with probability estimates |
| LDA           | Classification           | 79%      | Linear Discriminant Analysis           |
| Least Squares | Classification           | 75%      | Linear regression for classification   |
| PCA           | Dimensionality Reduction | N/A      | Principal Component Analysis           |
| SVD           | Matrix Factorization     | N/A      | Singular Value Decomposition           |

**Documentation:**

- Individual README files for each algorithm
- Mathematical explanations and code walkthroughs
- Output interpretation guides

**Data:**

- Cat/Dog training images (for classification algorithms)
- Test images (0.jpg to 9.jpg)
- Sample matrices (for PCA and SVD)

## How to Run

**Setup:**

```bash
git clone <repository-url>
cd MLLab
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
- Dependencies: numpy, opencv-python, scikit-learn, scipy

Each script runs independently and produces immediate results.

## 🔬 Technical Specifications

### **Data Processing**

- **Image Format**: JPEG, PNG, common formats
- **Preprocessing**: Grayscale conversion, resizing, normalization
- **Feature Extraction**: Pixel intensities as feature vectors
- **Data Split**: Training (cat/dog folders) vs. Testing (numbered files)

### **Algorithm Details**

- **SVM**: RBF kernel, probability estimates enabled
- **Bayesian**: Gaussian naive Bayes with Laplace smoothing
- **LDA**: Linear discriminant with covariance regularization
- **Least Squares**: Normal equation with sigmoid output
- **PCA**: Standardized features, 2-component projection
- **SVD**: Truncated decomposition, variance analysis

### **Performance Metrics**

- **Training Accuracy**: Percentage of correct training predictions
- **Confidence Scores**: Probability estimates for each prediction
- **Interpretability**: Feature importance and model parameters
- **Computational Efficiency**: Training and prediction time

## 🤝 Contributing

This project welcomes contributions! Areas for improvement:

- Additional algorithms (Random Forest, Neural Networks)
- More comprehensive datasets
- Performance optimization
- Extended documentation
- Visualization enhancements

## 📄 License

This project is designed for educational use. Feel free to use, modify, and distribute for learning purposes.

---

**Start your machine learning journey with clear, simple, and effective implementations that actually work!**
