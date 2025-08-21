import numpy as np
from sklearn.linear_model import LinearRegression
import joblib

# بيانات تدريب بسيطة (مساحة، عدد الغرف، عدد الحمامات) -> السعر
X = np.array([
    [100, 3, 1],
    [150, 4, 2],
    [200, 5, 2],
    [250, 6, 3],
    [300, 7, 3],
    [120, 3, 2],
    [180, 4, 2],
    [220, 5, 3],
])

y = np.array([100000, 150000, 200000, 250000, 300000, 130000, 180000, 230000])

# إنشاء النموذج وتدريبه
model = LinearRegression()
model.fit(X, y)

# حفظ النموذج
joblib.dump(model, "house_model.pkl")
print("✅ Model trained and saved as house_model.pkl")
