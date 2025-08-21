import os
from flask import Flask, request, jsonify, send_file
import joblib
import numpy as np

app = Flask(__name__)

# تحميل النموذج
model = joblib.load("house_model.pkl")

# الصفحة الرئيسية -> عرض index.html
@app.route("/")
def home():
    return send_file("index.html")

# التوقع
@app.route("/predict", methods=["POST"])
def predict():
    try:
        area = float(request.form["area"])
        rooms = int(request.form["rooms"])
        bathrooms = int(request.form["bathrooms"])

        features = np.array([[area, rooms, bathrooms]])
        prediction = model.predict(features)[0]

        return jsonify({"predicted_price": round(prediction, 2)})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
