import os
from flask import Flask, request, render_template_string
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("house_model.pkl")

# Load index.html as string
with open("index.html", "r") as f:
    index_html = f.read()

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        area = float(request.form["area"])
        rooms = int(request.form["rooms"])
        bathrooms = int(request.form["bathrooms"])

        input_data = np.array([[area, rooms, bathrooms]])
        prediction = model.predict(input_data)[0]
        prediction = round(prediction, 2)

    return render_template_string(index_html, prediction=prediction)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
