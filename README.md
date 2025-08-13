# Cats vs. Dogs Image Classification with SVM

## Overview

This repository contains a Python script for classifying images of cats and dogs from the Kaggle [Cats vs. Dogs dataset](https://www.kaggle.com/datasets/jakupymeraj/cats-and-dogs-image-dataset) using a Support Vector Machine (SVM). The model uses Histogram of Oriented Gradients (HOG) for feature extraction and a linear SVM for binary classification (cats vs. dogs).  

The dataset consists of 8,000 training images (4,000 cats, 4,000 dogs) and 2,000 test images (1,000 cats, 1,000 dogs), making it ideal for benchmarking image classification tasks.

The goal is to accurately classify images as either "cat" or "dog" and evaluate the model's performance using accuracy on the test set. The script also generates a CSV file with predictions for further analysis.

---

## Dataset

- **Source**: Kaggle Cats vs. Dogs dataset  
- **Structure**:
  - `training_set/`: 8,000 images (4,000 cats, 4,000 dogs)  
  - `test_set/`: 2,000 images (1,000 cats, 1,000 dogs)  
- **Format**: Color JPEG images in subdirectories `cats/` and `dogs/`  
- **Labels**: Binary (0 for cats, 1 for dogs)  

---

## Requirements

- Python 3.8+  
- Libraries:
  - pandas
  - numpy
  - opencv-python
  - scikit-learn
  - scikit-image
  - tqdm
  - joblib (optional for parallel processing)

# Cats vs. Dogs Image Classification with SVM

## Install Dependencies

Run the following command to install required Python libraries:

```bash
pip install pandas numpy opencv-python scikit-learn scikit-image tqdm joblib
````

---

## Usage

1. Download the Cats vs. Dogs dataset from Kaggle and extract it to:

```makefile
C:\Users\Admin\Desktop\task 2\cats_and_dogs\
```

2. Save the script as `svm_cats_dogs.py`.

3. Run the script:

```bash
cd C:\Users\Admin\Desktop\task 2
python svm_cats_dogs.py
```

The script will:

* Load and preprocess images
* Extract HOG features
* Train a linear SVM classifier
* Evaluate accuracy on the test set
* Save predictions to `cats_dogs_predictions.csv`

---

## Code Explanation

* **Load Images**: Reads images from `training_set/` and `test_set/` using OpenCV, resizes to 64×64 pixels, and converts to grayscale.
* **Feature Extraction**: Uses HOG (Histogram of Oriented Gradients) to extract edge and gradient features.
* **Preprocessing**: Scales features using `StandardScaler` for better SVM performance.
* **Model Training**: Trains a `LinearSVC` classifier (cats: 0, dogs: 1) with 5,000 iterations.
* **Evaluation**: Computes accuracy on the test set.
* **Output**: Saves predictions to `cats_dogs_predictions.csv` with columns:

  * `Filename`: Image file name
  * `True_Label`: Ground truth (cat or dog)
  * `Predicted_Label`: Predicted class (cat or dog)

---

## Model Choices

* **HOG Features**: Captures edge orientations, effective for distinguishing cats and dogs without deep learning frameworks.
* **Linear SVM**: Simple, efficient with high-dimensional HOG features, suitable for binary classification.
* **Preprocessing**: Images resized to 64×64 and converted to grayscale to reduce computational load. Features scaled for SVM convergence.

---

## Expected Output

**Console output:**

```text
Loading training data...
Loading test data...
Training SVM...
Test Accuracy: 0.XXXX
```

**File output:**

* `cats_dogs_predictions.csv`: Contains filenames, true labels, and predicted labels.

---

## Performance

* **Accuracy**: Typically 60–70% with HOG + Linear SVM, depending on dataset variability.
* **Limitations**: HOG may struggle with complex backgrounds or varied lighting. For higher accuracy, consider using a pretrained CNN (e.g., VGG16) for feature extraction.

---

## Potential Improvements

* **Feature Extraction**: Use pretrained CNN (VGG16, ResNet) for richer features with TensorFlow/PyTorch.
* **Hyperparameter Tuning**: Adjust SVM’s `C` parameter or try a kernel SVM (e.g., RBF) for better performance.
* **Cross-Validation**: Implement k-fold cross-validation for robust evaluation.
* **Data Augmentation**: Apply rotations, flips, or crops to training images to improve generalization.



## Dataset Notes

Ensure the dataset is extracted to:

```makefile
C:\Users\Admin\Desktop\task 2\cats_and_dogs\
```

with the correct subdirectory structure:
`training_set/cats/`, `training_set/dogs/`, `test_set/cats/`, `test_set/dogs/`.

If the dataset is elsewhere, update `train_dir` and `test_dir` in the script.

---

## License

MIT License. Feel free to use, modify, and distribute.

```
