import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Load dataset
data = pd.read_csv("model/data.csv")


# Convert caffeine values into numbers
data["caffeine"] = data["caffeine"].map({
    "None": 0,
    "Low": 1,
    "Moderate": 2,
    "High": 3
})


# Convert mood values into numbers
data["mood"] = data["mood"].map({
    "Anxious": 0,
    "Happy": 1,
    "Neutral": 2,
    "Sad": 3
})


# Convert sleep interruption values into numbers
data["sleep_interruptions"] = data["sleep_interruptions"].map({
    "No": 0,
    "Yes": 1
})


# Convert time into minutes
def time_to_minutes(time):
    hours, minutes = map(int, time.split(":"))
    return hours * 60 + minutes


data["bedtime"] = data["bedtime"].apply(time_to_minutes)
data["wake_up_time"] = data["wake_up_time"].apply(time_to_minutes)


# Separate input and output
X = data.drop("sleep_quality", axis=1)
y = data["sleep_quality"]


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Create Random Forest model
model = RandomForestClassifier(
    random_state=42
)


# Train model
model.fit(X_train, y_train)


# Make predictions
predictions = model.predict(X_test)


# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", accuracy)


# Save trained model
joblib.dump(model, "model/sleep_model.pkl")

print("Sleep Quality Model Saved Successfully!")