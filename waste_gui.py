import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import joblib

# Load the trained model
model = joblib.load("waste_classifier.pkl")  # Ensure this file exists

# Function to extract features (similar to how we trained the model)
def extract_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (128, 128))  # Resize to match training size
    features = image.flatten()  # Convert to 1D feature vector
    return np.array([features])  # Convert to 2D array for model prediction

# Function to predict waste category
def predict_waste(image_path):
    features = extract_features(image_path)
    prediction = model.predict(features)[0]
    return prediction

# Function to handle image upload
def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
    if file_path:
        display_image(file_path)
        classify_waste(file_path)

# Function to capture image from webcam
def capture_image():
    cap = cv2.VideoCapture(0)  # Open webcam
    ret, frame = cap.read()
    cap.release()  # Release webcam

    if ret:
        image_path = "captured_image.jpg"
        cv2.imwrite(image_path, frame)  # Save image
        display_image(image_path)
        classify_waste(image_path)

# Function to display image in GUI
def display_image(image_path):
    img = Image.open(image_path)
    img = img.resize((250, 250))  # Resize for display
    img = ImageTk.PhotoImage(img)
    panel.config(image=img)
    panel.image = img

# Function to classify and display result
def classify_waste(image_path):
    category = predict_waste(image_path)
    result_label.config(text=f"Predicted Category: {category}", fg="blue")

# GUI Setup
root = tk.Tk()
root.title("Waste Classification System")
root.geometry("500x600")

# Image Display Panel
panel = tk.Label(root)
panel.pack(pady=20)

# Buttons
upload_btn = tk.Button(root, text="📂 Upload Image", command=upload_image, font=("Arial", 14))
upload_btn.pack(pady=10)

capture_btn = tk.Button(root, text="📸 Capture from Webcam", command=capture_image, font=("Arial", 14))
capture_btn.pack(pady=10)

# Prediction Result Label
result_label = tk.Label(root, text="Prediction: ", font=("Arial", 16, "bold"))
result_label.pack(pady=20)

# Run the GUI
root.mainloop()
