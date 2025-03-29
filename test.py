import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import joblib

# Load dataset
data = pd.read_csv("features.csv")

# Extract features (X) and labels (y)
X = data.iloc[:, 2:].values
y = data.iloc[:, 1].values

# Extract unique class labels
unique_classes = np.unique(y)  # ✅ Fix: Define unique_classes

# Compute class weights
class_weights = compute_class_weight(class_weight="balanced", classes=unique_classes, y=y)
class_weight_dict = {cls: weight for cls, weight in zip(unique_classes, class_weights)}

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=200, class_weight=class_weight_dict, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy * 100:.2f}%")
print("\n🔍 Classification Report:\n", classification_report(y_test, y_pred))

# Save model & label classes
joblib.dump(model, "waste_classifier.pkl")
print("🎯 Model trained & saved as 'waste_classifier.pkl'")

joblib.dump(unique_classes, "label_classes.pkl")  # ✅ Fix: Save label classes
print("📁 Label classes saved as 'label_classes.pkl'")
