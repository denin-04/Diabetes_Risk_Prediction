
from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)
app.secret_key = "diabetes_prediction_project_key"


# ==========================================
# LOAD MACHINE LEARNING FILES
# ==========================================

models = joblib.load("models/all_models.pkl")
imputer = joblib.load("models/imputer.pkl")
scaler = joblib.load("models/scaler.pkl")

comparison = pd.read_csv("models/model_comparison.csv")


# ==========================================
# HOME PAGE - OPEN PREDICTION FIRST
# ==========================================

@app.route("/")
def home():
    return redirect(url_for("predict"))


# ==========================================
# DASHBOARD PAGE
# ==========================================

@app.route("/dashboard")
def dashboard():

    best_algorithm = comparison.iloc[0]["Algorithm"]
    best_accuracy = comparison.iloc[0]["Accuracy"]

    return render_template(
        "index.html",
        comparison=comparison.to_dict(orient="records"),
        best_algorithm=best_algorithm,
        best_accuracy=round(best_accuracy, 2)
    )


# ==========================================
# PREDICTION PAGE
# ==========================================

@app.route("/predict", methods=["GET", "POST"])
def predict():

    if request.method == "POST":

        try:

            # Get patient input values
            input_data = pd.DataFrame([{
                "Pregnancies": float(request.form["pregnancies"]),
                "Glucose": float(request.form["glucose"]),
                "BloodPressure": float(request.form["blood_pressure"]),
                "SkinThickness": float(request.form["skin_thickness"]),
                "Insulin": float(request.form["insulin"]),
                "BMI": float(request.form["bmi"]),
                "DiabetesPedigreeFunction": float(request.form["dpf"]),
                "Age": float(request.form["age"])
            }])

            # Save original values
            session["patient_values"] = {
                key: float(value)
                for key, value in input_data.iloc[0].to_dict().items()
            }

            # Replace zero values with missing values
            columns_with_zero = [
                "Glucose",
                "BloodPressure",
                "SkinThickness",
                "Insulin",
                "BMI"
            ]

            input_data[columns_with_zero] = (
                input_data[columns_with_zero].replace(0, np.nan)
            )

            # Apply preprocessing
            processed_data = imputer.transform(input_data)
            processed_data = scaler.transform(processed_data)

            # Get predictions from all algorithms
            prediction_results = []

            for name, algorithm in models.items():

                prediction = algorithm.predict(processed_data)[0]

                if prediction == 1:
                    result = "Higher Risk"
                else:
                    result = "Lower Risk"

                accuracy = comparison.loc[
                    comparison["Algorithm"] == name,
                    "Accuracy"
                ].iloc[0]

                prediction_results.append({
                    "name": name,
                    "prediction": result,
                    "accuracy": round(float(accuracy), 2)
                })

            # Majority voting
            higher_count = sum(
                1 for item in prediction_results
                if item["prediction"] == "Higher Risk"
            )

            lower_count = len(prediction_results) - higher_count

            if higher_count > lower_count:
                overall_result = "High Risk"
            else:
                overall_result = "Low Risk"

            # Model agreement percentage
            overall_percentage = (
                max(higher_count, lower_count)
                / len(prediction_results)
            ) * 100

            # Save results in session
            session["prediction_results"] = prediction_results
            session["overall_result"] = overall_result
            session["higher_count"] = higher_count
            session["lower_count"] = lower_count
            session["overall_percentage"] = round(
                overall_percentage, 1
            )

            return redirect(url_for("results"))

        except (ValueError, KeyError):
            return render_template(
                "predict.html",
                error="Please enter valid values in all fields."
            )

    return render_template("predict.html")


# ==========================================
# RESULTS PAGE
# ==========================================

@app.route("/results")
def results():

    prediction_results = session.get("prediction_results")
    patient_values = session.get("patient_values")

    if not prediction_results or not patient_values:
        return redirect(url_for("predict"))

    return render_template(
        "results.html",
        prediction_results=prediction_results,
        patient_values=patient_values,
        overall_result=session.get("overall_result"),
        higher_count=session.get("higher_count"),
        lower_count=session.get("lower_count"),
        overall_percentage=session.get("overall_percentage")
    )


# ==========================================
# SUGGESTIONS PAGE
# ==========================================

@app.route("/suggestions")
def suggestions():
    return render_template("suggestions.html")


# ==========================================
# ABOUT PAGE
# ==========================================

@app.route("/about")
def about():
    return render_template("about.html")


# ==========================================
# RUN APPLICATION
# ==========================================

if __name__ == "__main__":
    app.run(debug=True)
