import cv2
import numpy as np
import pandas as pd
import os

# Path to dataset
dataset_folder = "dataset_processed"

# Function to extract features from an image
def extract_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None

    image = cv2.resize(image, (32, 32))  # Resize to 32x32
    features = image.flatten()  # Flatten to 1024 features
    return features

# Process all images in dataset
feature_data = []
for category in os.listdir(dataset_folder):
    category_path = os.path.join(dataset_folder, category)
    if not os.path.isdir(category_path):
        continue

    for filename in os.listdir(category_path):
        image_path = os.path.join(category_path, filename)
        features = extract_features(image_path)
        if features is not None:
            feature_data.append([filename, category] + list(features))

# Convert to DataFrame and save
columns = ["Filename", "Label"] + [f"Feature_{i}" for i in range(len(features))]
df = pd.DataFrame(feature_data, columns=columns)
df.to_csv("features.csv", index=False)

print("✅ Feature extraction completed! Saved as features.csv")
