import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import joblib

# Load extracted features
data = pd.read_csv("features.csv")

# Check dataset structure
print("📊 Dataset preview:\n", data.head())

# Extract features and labels
X = data.iloc[:, 2:].values  # Feature columns
y = data.iloc[:, 1].values   # Label column

# Print class distribution
unique_classes, counts = np.unique(y, return_counts=True)
print("🛠 Class Distribution:", dict(zip(unique_classes, counts)))

# Compute class weights for handling imbalanced data
class_weights = compute_class_weight(class_weight="balanced", classes=unique_classes, y=y)
class_weight_dict = {cls: weight for cls, weight in zip(unique_classes, class_weights)}

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train model with class weights
model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight=class_weight_dict)
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy * 100:.2f}%")
print("\n🔍 Classification Report:\n", classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "waste_classifier.pkl")
print("🎯 Model trained & saved as 'waste_classifier.pkl'")

# Save class labels (for decoding predictions)
joblib.dump(unique_classes, "label_classes.pkl")
print("📁 Label classes saved as 'label_classes.pkl'")
