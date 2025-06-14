import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load dataset
df = pd.read_csv("gender_voice_dataset.csv")

# Encode labels
le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])  # male=1, female=0

# Split features and labels
X = df.drop('label', axis=1)
y = df['label']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train SVM model
model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("\n📉 Confusion Matrix:\n", cm)

# Save model and scaler
with open("gender_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ Model saved as gender_model.pkl")
print("✅ Scaler saved as scaler.pkl")
