import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
data = pd.read_csv("dataset.csv", header=None)

# Separate features and labels
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Split training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = RandomForestClassifier()

# Train model
model.fit(X_train, y_train)

# Test accuracy
accuracy = model.score(X_test, y_test)

print("Model Accuracy:", accuracy)

# Save trained model
joblib.dump(model, "gesture_model.pkl")

print("Model saved as gesture_model.pkl")