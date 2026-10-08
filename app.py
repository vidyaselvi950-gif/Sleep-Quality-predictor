from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load trained model
model = joblib.load("model/sleep_model.pkl")

# File to store prediction history
HISTORY_FILE = "prediction_history.csv"


# Convert time into minutes
def time_to_minutes(time):
    hours, minutes = map(int, time.split(":"))
    return hours * 60 + minutes


# Load previous history
def load_history():
    if os.path.exists(HISTORY_FILE):
        data = pd.read_csv(HISTORY_FILE)
        return data.to_dict("records")

    return []


@app.route("/")
def home():

    history = load_history()

    return render_template(
        "index.html",
        history=history
    )


@app.route("/predict", methods=["POST"])
def predict():

    sleep_duration = float(request.form["sleep_duration"])
    exercise_duration = float(request.form["exercise_duration"])
    screen_time = float(request.form["screen_time"])

    caffeine = int(request.form["caffeine"])
    stress_level = float(request.form["stress_level"])
    mood = int(request.form["mood"])
    sleep_interruptions = int(request.form["sleep_interruptions"])

    bedtime = time_to_minutes(request.form["bedtime"])
    wake_up_time = time_to_minutes(request.form["wake_up_time"])

    features = [[
        sleep_duration,
        exercise_duration,
        screen_time,
        caffeine,
        stress_level,
        mood,
        sleep_interruptions,
        bedtime,
        wake_up_time
    ]]

    prediction = model.predict(features)[0]

    # Save prediction history
    new_record = pd.DataFrame([{
        "sleep_duration": sleep_duration,
        "bedtime": request.form["bedtime"],
        "wake_up_time": request.form["wake_up_time"],
        "quality": prediction
    }])

    new_record.to_csv(
        HISTORY_FILE,
        mode="a",
        header=not os.path.exists(HISTORY_FILE),
        index=False
    )

    history = load_history()

    return render_template(
        "index.html",
        prediction=prediction,
        history=history
    )


if __name__ == "__main__":
    app.run(debug=True)