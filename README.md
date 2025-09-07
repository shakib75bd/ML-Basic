# ML-Basic: Simple Machine Learning Implementations for Binary Classification

A comprehensive collection of **simplified machine learning algorithms** for educational purposes and practical binary classification. This project demonstrates core ML concepts through clean, readable implementations that prioritize **understanding over complexity**.

## 🎯 What This Project Has

### **6 Complete Machine Learning Algorithms**

| Algorithm                              | Accuracy | Type                     | Key Feature                               |
| -------------------------------------- | -------- | ------------------------ | ----------------------------------------- |
| **SVM** (Support Vector Machine)       | 100%     | Kernel-based             | Maximum margin classification             |
| **Bayesian** (Naive Bayes)             | 92.9%    | Probabilistic            | Uncertainty quantification                |
| **LDA** (Linear Discriminant Analysis) | 79%      | Statistical              | Dimensionality reduction + classification |
| **Least Squares**                      | 75%      | Regression-based         | Closed-form solution                      |
| **PCA** (Principal Component Analysis) | N/A      | Dimensionality reduction | Variance maximization                     |
| **SVD** (Singular Value Decomposition) | N/A      | Matrix factorization     | Low-rank approximation                    |

### **Complete Documentation Suite**

- **Individual README files** for each algorithm with mathematical foundations
- **Code walkthroughs** explaining every step
- **Output interpretation** with real performance metrics
- **Comparison tables** showing strengths and weaknesses
- **Practical applications** and use cases

### **Binary Classification Dataset**

- **Training Data**: Cat and Dog images organized in folders
- **Test Data**: 10 numbered test images (0.jpg to 9.jpg)
- **Preprocessing**: Automatic 8x8 or 16x16 grayscale conversion
- **Normalization**: Pixel values scaled to [0,1] range

## 🚀 How to Run These Codes

### **Quick Start**

1. **Clone and Setup**

```bash
git clone <repository-url>
cd MLLab
pip install -r requirements.txt
```

2. **Run Any Algorithm**

```bash
python svm.py        # Support Vector Machine
python bayesian.py   # Bayesian Classification
python lda.py        # Linear Discriminant Analysis
python least_square.py # Least Squares Classification
python pca.py        # Principal Component Analysis
python svd.py        # Singular Value Decomposition
```

3. **Test with Different Images**

- Change the test image by modifying the filename in any script
- Available test images: `0.jpg`, `1.jpg`, `2.jpg`, ..., `9.jpg`
- Add your own images to the `TestData/` folder

### **System Requirements**

- **Python**: 3.7 or higher
- **Operating System**: Windows, macOS, or Linux
- **Memory**: Minimal (all algorithms use small datasets)
- **Dependencies**: See `requirements.txt`

### **File Structure**

```
MLLab/
├── svm.py                 # Support Vector Machine implementation
├── svm_readme.md          # SVM theory and documentation
├── bayesian.py            # Bayesian classifier implementation
├── bayesian_readme.md     # Bayesian theory and documentation
├── lda.py                 # Linear Discriminant Analysis
├── lda_readme.md          # LDA theory and documentation
├── least_square.py        # Least squares classifier
├── least_square_readme.md # Least squares theory and documentation
├── pca.py                 # Principal Component Analysis
├── pca_readme.md          # PCA theory and documentation
├── svd.py                 # Singular Value Decomposition
├── svd_readme.md          # SVD theory and documentation
├── requirements.txt       # Python dependencies
├── Training Data/         # Cat and Dog training images
│   ├── Cat/              # Cat training images
│   └── Dog/              # Dog training images
└── TestData/             # Test images (0.jpg to 9.jpg)
```

### **Customization Options**

- **Change dataset size**: Modify the slice `[:10]` in any script to use more/fewer training images
- **Adjust image resolution**: Change `(8, 8)` to `(16, 16)` or other sizes
- **Test different images**: Replace test image path in scripts
- **Modify parameters**: Tune hyperparameters in each algorithm

## ✨ Benefits of These Simplest Codes

### **1. Educational Excellence**

- **Clear Learning Path**: Start with simple concepts, build to complex ones
- **Mathematical Foundation**: Each algorithm includes theoretical background
- **Step-by-Step Explanation**: Every code section is documented and explained
- **Visual Results**: See exactly what each algorithm produces
- **Comparative Analysis**: Understand trade-offs between different approaches

### **2. Practical Advantages**

#### **🔧 Simplicity**

- **Minimal Dependencies**: Only essential libraries (NumPy, OpenCV, scikit-learn)
- **Short Code**: Each implementation is 30-50 lines, easy to understand
- **No Complex Setup**: Works out-of-the-box with minimal configuration
- **Quick Execution**: All algorithms run in seconds

#### **⚡ Performance**

- **Fast Training**: Most algorithms train instantly
- **Low Memory**: Works on any modern computer
- **Real-time Inference**: Suitable for interactive applications
- **Scalable**: Easy to extend to larger datasets

#### **🎓 Learning Benefits**

- **Hands-on Experience**: Actually implement ML from scratch
- **Algorithm Comparison**: See how different approaches work
- **Parameter Understanding**: Learn what each setting does
- **Debug Friendly**: Simple enough to trace through every step

### **3. Production Readiness**

#### **✅ Robust Implementation**

- **Error Handling**: Graceful handling of missing files and edge cases
- **Numerical Stability**: Proper handling of edge cases and numerical precision
- **Consistent Interface**: Same pattern across all algorithms
- **Modular Design**: Easy to integrate into larger projects

#### **📊 Comprehensive Output**

- **Performance Metrics**: Training accuracy, confidence scores
- **Probability Estimates**: Not just predictions but confidence levels
- **Interpretable Results**: Clear explanation of what each number means
- **Debug Information**: Detailed intermediate results for troubleshooting

### **4. Real-World Applications**

#### **🏥 Academic Research**

- **Baseline Models**: Use as comparison for complex algorithms
- **Teaching Tool**: Demonstrate ML concepts to students
- **Rapid Prototyping**: Quick validation of ideas
- **Algorithm Study**: Understand fundamental differences between methods

#### **🏢 Industry Use**

- **Proof of Concept**: Validate ML feasibility for business problems
- **Embedded Systems**: Deploy on resource-constrained devices
- **Edge Computing**: Run classification on mobile/IoT devices
- **Baseline Performance**: Establish minimum viable model performance

#### **🔬 Research Benefits**

- **Feature Engineering**: Test which features matter most
- **Data Quality Assessment**: Understand if data is sufficient
- **Algorithm Selection**: Choose best method for specific problems
- **Hyperparameter Sensitivity**: See how parameters affect performance

### **5. Extensibility**

#### **🔄 Easy Modifications**

- **Add New Features**: Extend feature extraction
- **Change Preprocessing**: Modify image processing pipeline
- **Tune Parameters**: Experiment with different settings
- **Hybrid Approaches**: Combine multiple algorithms

#### **📈 Scalability Path**

- **Larger Datasets**: Increase training data size
- **More Classes**: Extend to multi-class classification
- **Complex Features**: Add sophisticated feature engineering
- **Deep Learning**: Bridge to neural network implementations

## 🎯 Why Choose Simple Over Complex?

### **Understanding First**

- **Foundation Building**: Master fundamentals before advanced techniques
- **Debugging Skills**: Learn to troubleshoot when things go wrong
- **Parameter Intuition**: Understand what each setting actually does
- **Algorithm Comparison**: See strengths and weaknesses clearly

### **Practical Benefits**

- **Faster Development**: Get results quickly without complex setup
- **Lower Risk**: Simple systems are more reliable and predictable
- **Easier Maintenance**: Code that anyone can understand and modify
- **Resource Efficiency**: Run anywhere without special hardware

### **Learning Outcomes**

- **Mathematical Insight**: See the math behind each algorithm
- **Implementation Skills**: Learn to code ML from first principles
- **Problem-Solving**: Understand when to use which algorithm
- **Performance Analysis**: Interpret results and make improvements

## 📚 Learning Path Recommendation

### **Beginner**: Start Here

1. **Least Squares** (`least_square.py`) - Simplest linear approach
2. **Bayesian** (`bayesian.py`) - Probabilistic reasoning
3. **LDA** (`lda.py`) - Statistical classification

### **Intermediate**: Build Understanding

4. **PCA** (`pca.py`) - Dimensionality reduction concepts
5. **SVD** (`svd.py`) - Matrix factorization fundamentals
6. **SVM** (`svm.py`) - Advanced kernel methods

### **Advanced**: Comparative Analysis

- Run all algorithms on same data
- Compare performance and interpretability
- Understand trade-offs and use cases
- Experiment with different parameters

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
