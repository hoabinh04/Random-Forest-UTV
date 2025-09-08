from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np

# Đọc dữ liệu
df = pd.read_csv("data.csv")
df = df.drop(columns=["id", "Unnamed: 32"])  # bỏ cột không cần thiết

# Label
y = df["diagnosis"].map({"M": 1, "B": 0})
X = df[["area_worst", "concave points_worst", "concave points_mean",
        "radius_worst", "concavity_mean"]]

# Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Accuracy
accuracy = model.score(X_test, y_test)
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            # Lấy dữ liệu từ form
            features = [float(x) for x in request.form.values()]
            features = np.array(features).reshape(1, -1)
            
            # Dự đoán
            pred = model.predict(features)[0]
            prediction = "Malignant (Ác tính)" if pred == 1 else "Benign (Lành tính)"
        except:
            prediction = "⚠️ Lỗi: Vui lòng nhập đủ 5 giá trị hợp lệ."

    return render_template("index.html", prediction=prediction, accuracy=round(accuracy*100, 2))

if __name__ == "__main__":
    app.run(debug=True)
