import os
import joblib
import base64
import numpy as np
from flask import Flask, request, render_template, jsonify, url_for
from werkzeug.utils import secure_filename
from PIL import Image
from io import BytesIO

# Set the absolute path to the templates folder
TEMPLATE_DIR = r"C:\Users\Manas Deshmukh\OneDrive\Desktop\Mac_DIP Project - Copy\templates"

app = Flask(__name__, template_folder=TEMPLATE_DIR)

# Configure Upload Folder
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load Model
MODEL_PATH = "waste_classifier.pkl"
LABEL_ENCODER_PATH = "label_encoder.pkl"

model = None
label_encoder = None

if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        if not hasattr(model, "predict"):
            raise ValueError("⚠️ Model does not have a `predict()` method!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        model = None
else:
    print("❌ Model file 'waste_classifier.pkl' not found!")

# Load Label Encoder (if available)
if os.path.exists(LABEL_ENCODER_PATH):
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
else:
    print("⚠️ Label encoder file not found. Using default CATEGORY_MAPPING.")

# Default Category Mapping (if LabelEncoder is missing)
CATEGORY_MAPPING = {
    0: "Paper",
    1: "Cardboard",
    2: "Metal",
    3: "Glass",
    4: "Plastic",
    5: "Organic Waste",
    6: "E-Waste"
}

# Preprocessing Function
def preprocess_image(image_path=None, image_data=None, target_size=(32, 32)):
    """Convert image to grayscale, resize, and flatten for model input."""
    try:
        if image_path:
            img = Image.open(image_path).convert("L")  # Convert to grayscale
        elif image_data:
            img = Image.open(BytesIO(image_data)).convert("L")  # Convert Base64 to image
        else:
            return None

        img = img.resize(target_size)  # Resize image
        img_array = np.array(img).flatten()  # Flatten to 1D array
        return img_array.reshape(1, -1)  # Reshape for model
    except Exception as e:
        print(f"❌ Error processing image: {e}")
        return None

# Home Route
@app.route("/", methods=["GET", "POST"])
def home():
    uploaded_image_path = None
    predicted_category = None

    if request.method == "POST":
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded!"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No selected file!"}), 400

        # Save Uploaded File
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)
        uploaded_image_path = url_for("static", filename=f"uploads/{filename}")

        # Preprocess Image
        features = preprocess_image(image_path=file_path)
        if features is None:
            return jsonify({"error": "Image processing failed!"}), 500

        # Make Prediction
        if model:
            prediction = model.predict(features)[0]

            # Convert Prediction to Label
            if isinstance(prediction, str):
                predicted_category = prediction  # If model returns string labels
            elif label_encoder:
                predicted_category = label_encoder.inverse_transform([prediction])[0]  # Decode integer labels
            else:
                predicted_category = CATEGORY_MAPPING.get(int(prediction), "Unknown")  # Use default mapping

        else:
            predicted_category = "❌ Model not loaded properly!"

    return render_template("index.html", prediction=predicted_category, image_path=uploaded_image_path)

# Run Flask App
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
