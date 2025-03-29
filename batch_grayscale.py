import cv2
import os

# Input and Output Folder Paths
input_folder = r"C:\Users\Manas Deshmukh\OneDrive\Desktop\Mac_DIP Project\dataset"
output_folder = r"C:\Users\Manas Deshmukh\OneDrive\Desktop\Mac_DIP Project\dataset_processed"

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Process all images in subfolders
for root, dirs, files in os.walk(input_folder):
    for filename in files:
        if filename.endswith(".jpg") or filename.endswith(".png"):  # Add more extensions if needed
            image_path = os.path.join(root, filename)

            # Create matching subfolder in output
            relative_path = os.path.relpath(root, input_folder)
            save_folder = os.path.join(output_folder, relative_path)
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)

            output_path = os.path.join(save_folder, filename)

            # Read and Convert to Grayscale
            image = cv2.imread(image_path)
            if image is None:
                print(f"⚠️ Could not read {filename}, skipping...")
                continue

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Save the grayscale image
            cv2.imwrite(output_path, gray)
            print(f"✅ Converted: {filename} -> {output_path}")

print("✅ All images converted successfully!")
