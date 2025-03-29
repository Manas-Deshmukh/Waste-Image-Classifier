import cv2
import os
import numpy as np

# Input and Output Folder Paths
input_folder = r"C:\Users\Manas Deshmukh\OneDrive\Desktop\Mac_DIP Project\dataset_grayscale"
output_folder = r"C:\Users\Manas Deshmukh\OneDrive\Desktop\Mac_DIP Project\dataset_processed"

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Process all images in the folder
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # Add more extensions if needed
        image_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # Read the image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Could not read {filename}, skipping...")
            continue

        # Resize image to 128x128 (Change size if needed)
        resized = cv2.resize(image, (128, 128))

        # Apply Gaussian Blur (Noise Reduction)
        denoised = cv2.GaussianBlur(resized, (3, 3), 0)

        # Apply Histogram Equalization (Contrast Enhancement)
        enhanced = cv2.equalizeHist(denoised)

        # Apply Edge Detection (Canny Edge)
        edges = cv2.Canny(enhanced, 50, 150)

        # Save the preprocessed image
        cv2.imwrite(output_path, edges)
        print(f"Processed: {filename} -> {output_path}")

print("✅ All images preprocessed successfully!")
