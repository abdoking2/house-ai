import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Sample dataset (can be replaced with real estate dataset)
data = {
    "area": [50, 60, 80, 100, 120, 150],
    "rooms": [2, 2, 3, 3, 4, 5],
    "bathrooms": [1, 1, 2, 2, 2, 3],
    "price": [50000, 60000, 90000, 110000, 150000, 200000]
}

df = pd.DataFrame(data)

X = df[["area", "rooms", "bathrooms"]]
y = df["price"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
joblib.dump(model, "house_model.pkl")
print("Model trained and saved successfully!")
