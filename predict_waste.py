import cv2
import numpy as np
import joblib

# Load trained model
model = joblib.load("waste_classifier.pkl")

# Function to extract features for prediction
def extract_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("⚠️ Error: Could not read image!")
        return None

    image = cv2.resize(image, (32, 32))  # Match training size
    features = image.flatten()  # Convert image into a feature vector (1024 features)
    return features

# Predict category of a new image
def predict_category(image_path):
    features = extract_features(image_path)
    if features is None:
        return

    features = np.array(features).reshape(1, -1)  # Ensure correct input shape
    prediction = model.predict(features)[0]

    print(f"🔍 Predicted Category: {prediction}")

# Example Usage
image_path = "dataset_processed/plastic/plastic1.jpg"
predict_category(image_path)
