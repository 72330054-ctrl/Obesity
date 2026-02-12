from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
app = Flask(__name__)
import requests

# ===============================
# Load models + scaler + encoders
# ===============================
with open("models.pkl", "rb") as f:
    artifacts = pickle.load(f)

models = artifacts["models"]
feature_columns = artifacts["feature_columns"]
scaler = artifacts["scaler"]
label_encoders = artifacts["label_encoders"]

# ===============================
# Formula Risk Score (Your logic)
# ===============================
def calculate_formula_risk(form):
    age = int(form['age'])
    exercise = int(form['exercise'])
    technology = int(form['technology'])
    transport = int(form['transport'])

    height_cm = float(form["height"])
    weight = float(form["weight"])
    height_m = height_cm / 100
    bmi = weight / (height_m ** 2)

    fastfood = int(form.get('fastfood', 0))
    smoking = int(form.get('smoking', 0))
    calories = int(form.get('calories', 0))
    family_history = int(form.get('family_history', 0))
    veggies = int(form['vegetable'])
    liquid = int(form['liquid'])
    snacks = int(form['snacks'])
    mainmeals = int(form['mainmeals'])
    alcohol = int(form['alcohol'])

    risk = 0

    # BMI
    if 25 <= bmi < 30: risk += 10
    elif bmi >= 30: risk += 15

    # Age + family history
    if 18 <= age < 26 and family_history == 1: risk += 25
    elif 26 <= age < 33 and family_history == 1: risk += 22
    elif 33 <= age < 41 and family_history == 1: risk += 18
    elif 41 <= age < 51 and family_history == 1: risk += 15

    if fastfood == 1: risk += 20
    if smoking == 0: risk += 2
    if calories == 1: risk -= 3

    if liquid == 1: risk += 3
    elif liquid == 2: risk += 1

    if veggies == 1: risk += 3
    elif veggies == 2: risk += 1

    if mainmeals == 2: risk += 1
    elif mainmeals == 3: risk += 3

    if snacks == 2: risk += 2
    elif snacks == 3: risk += 5
    elif snacks == 4: risk += 7

    if alcohol == 1: risk += 1
    elif alcohol == 2: risk += 3
    elif alcohol == 3: risk += 5

    if exercise == 0: risk += 8
    elif exercise == 1: risk += 3
    elif exercise == 2: risk -= 3
    else: risk -= 8

    if technology == 1: risk += 2
    elif technology == 2: risk += 3

    if transport in [0,1]: risk += 5
    elif transport == 2: risk -= 5
    elif transport == 3: risk += 3
    elif transport == 4: risk -= 5

    return round(risk, 2), round(bmi, 2)



# ===============================
# Main Route
# ===============================
@app.route("/", methods=["GET", "POST"])
def index():
    formula_risk = None
    bmi = None
    ml_results = {}
    ai_recommendation = None   

    if request.method == "POST":
        form = request.form
        print("Received form data:", form) 

        # -----------------------------
        # Read values
        # -----------------------------
        age = int(form["age"])
        gender_text = form["gender"]   # male / female
        height_cm = float(form["height"])
        weight = float(form["weight"])

        exercise = int(form["exercise"])
        technology = int(form["technology"])
        transport = int(form["transport"])
        fastfood = int(form.get("fastfood", 0))
        smoking = int(form.get("smoking", 0))
        calories = int(form.get("calories", 0))
        family_history = int(form.get("family_history", 0))
        veggies = int(form["vegetable"])
        liquid = int(form["liquid"])
        snacks = int(form["snacks"])
        mainmeals = int(form["mainmeals"])
        alcohol = int(form["alcohol"])
        
        # -----------------------------
        # BMI
        # -----------------------------
        height_m = height_cm / 100
        bmi = weight / (height_m * height_m)

        # -----------------------------
        # Encode gender correctly
        # -----------------------------
        gender = label_encoders["Gender"].transform([gender_text])[0]

        # -----------------------------
        # Formula score
        # -----------------------------
        formula_risk, bmi = calculate_formula_risk(form)

        # -----------------------------
        # Build ML input (by column names)
        # -----------------------------
        input_dict = {
            "Gender": gender,
            "Age": age,
            "BMI": bmi,
            "family_history_with_overweight": family_history,
            "high-caloric-food-frequently": fastfood,
            "include vegetables in your meals": veggies,
            "nb-main-meals-daily": mainmeals,
            "snacks-between-meals": snacks,
            "SMOKE": smoking,
            "water-daily": liquid,
            "monitor-calory-intake": calories,
            "physical-activity": exercise,
            "technology-time": technology,
            "alcohol": alcohol,
            "transportation": transport
        }

        input_df = pd.DataFrame([input_dict])
        input_df = input_df[feature_columns]  # correct order
        X_scaled = scaler.transform(input_df)

        # -----------------------------
        # Predict with all models
        # -----------------------------
        for name, model in models.items():
            pred = model.predict(X_scaled)[0]
            ml_results[name] = round(float(pred), 3)
        # -----------------------------
        # Get AI recommendation
        # -----------------------------
       # -----------------------------

    return render_template(
        "index.html",
        formula_risk=formula_risk,
        bmi=bmi,
        ml_results=ml_results,
        
    )


if __name__ == "__main__":
    app.run(debug=True)











# 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