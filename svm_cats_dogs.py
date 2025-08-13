import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import glob
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

# Define paths to dataset
train_dir = r'C:\Users\Admin\Desktop\task 3\dataset\train_set'
test_dir = r'C:\Users\Admin\Desktop\task 3\dataset\test_set'

# Function to extract HOG features for a single image
def extract_hog_features(img_path, label_map, class_name, img_size=(64, 64)):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None
    img = cv2.resize(img, img_size)
    hog_features = hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
    label = label_map[class_name]
    return hog_features, label

# Function to load images and extract HOG features (parallel)
def load_and_extract_features(directory, label_map={'cats': 0, 'dogs': 1}, img_size=(64, 64)):
    features = []
    labels = []
    for class_name in ['cats', 'dogs']:
        class_dir = os.path.join(directory, class_name)
        img_paths = glob.glob(os.path.join(class_dir, '*.jpg'))

        # Parallel HOG extraction with tqdm progress bar
        results = Parallel(n_jobs=-1)(delayed(extract_hog_features)(p, label_map, class_name, img_size) for p in tqdm(img_paths, desc=f"Processing {class_name}"))
        
        # Filter out any None results
        for feat, label in results:
            if feat is not None:
                features.append(feat)
                labels.append(label)
    return np.array(features), np.array(labels)

# Load training and test data
print("Loading training data...")
X_train, y_train = load_and_extract_features(train_dir)
print("Loading test data...")
X_test, y_test = load_and_extract_features(test_dir)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Linear SVM
print("Training SVM...")
svm = LinearSVC(random_state=42, max_iter=5000)  # Increased max_iter for safety
svm.fit(X_train_scaled, y_train)

# Predict on test set
y_pred = svm.predict(X_test_scaled)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")

# Save predictions to CSV
test_files = []
test_labels = []
for class_name in ['cats', 'dogs']:
    class_dir = os.path.join(test_dir, class_name)
    test_files.extend([os.path.basename(f) for f in glob.glob(os.path.join(class_dir, '*.jpg'))])
    test_labels.extend([class_name] * len(glob.glob(os.path.join(class_dir, '*.jpg'))))

# Create DataFrame with predictions
pred_df = pd.DataFrame({
    'Filename': test_files,
    'True_Label': test_labels,
    'Predicted_Label': np.where(y_pred == 0, 'cat', 'dog')
})
pred_df.to_csv('cats_dogs_predictions.csv', index=False)
print("Predictions saved to 'cats_dogs_predictions.csv'")
