from flask import Flask, render_template, request
from src.components.pipeline.predict_pipeline import CustomData, PredictionPipeline

# Create Flask app
app = Flask(__name__)

#  Home page route
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")  # This loads the home page

# Predict route
@app.route("/predict", methods=["GET", "POST"])
def predict():
    prediction = None
    proba = None

    if request.method == "POST":
        data = CustomData(
            gender=request.form["gender"],
            SeniorCitizen=int(request.form["SeniorCitizen"]),
            Partner=request.form["Partner"],
            Dependents=request.form["Dependents"],
            tenure=int(request.form["tenure"]),
            PhoneService=request.form["PhoneService"],
            MultipleLines=request.form["MultipleLines"],
            InternetService=request.form["InternetService"],
            OnlineSecurity=request.form["OnlineSecurity"],
            OnlineBackup=request.form["OnlineBackup"],
            DeviceProtection=request.form["DeviceProtection"],
            TechSupport=request.form["TechSupport"],
            StreamingTV=request.form["StreamingTV"],
            StreamingMovies=request.form["StreamingMovies"],
            Contract=request.form["Contract"],
            PaperlessBilling=request.form["PaperlessBilling"],
            PaymentMethod=request.form["PaymentMethod"],
            MonthlyCharges=float(request.form["MonthlyCharges"]),
            TotalCharges=float(request.form["TotalCharges"])
        )

        df = data.get_data_as_data_frame()
        pipeline = PredictionPipeline()
        proba = pipeline.predict_proba(df)[0][1]
        prediction = "Churn" if proba >= 0.4 else "No Churn"

    # Render predict.html whether GET or POST
    return render_template("predict.html", prediction=prediction, proba=proba)

# Run app
if __name__ == "__main__":
    app.run(debug=True)
