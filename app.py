from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
import joblib

app = Flask(__name__)

# Nutritional data for food detection
nutritional_info = {
    'Fast Food': {1: {"name": "Hamburger", "carbs": 30, "calories": 250, "alternative": "Grilled Chicken"},
                  2: {"name": "Pizza", "carbs": 36, "calories": 285, "alternative": "Whole Wheat Wrap"},
                  0: {"name": "French fries", "carbs": 35, "calories": 365, "alternative": "Baked Sweet Potato"}},
    'Vegetables': {6: {"name": "Watermelon", "carbs": 8, "calories": 30},
                   5: {"name": "Tomato", "carbs": 3.9, "calories": 18},
                   4: {"name": "Carrot", "carbs": 10, "calories": 41}},
    'Fruit': {6: {"name": "Watermelon", "carbs": 8, "calories": 30},
              0: {"name": "Banana", "carbs": 27, "calories": 105, "alternative": "Apple"},
              1: {"name": "Mango", "carbs": 25, "calories": 99, "alternative": "Peach"}},
    'Product': {2: {"name": "Pepsi", "carbs": 28, "calories": 150, "alternative": "Fresh Juice"},
                5: {"name": "Hohos", "carbs": 35, "calories": 240, "alternative": "Protein Bar"}}
}

model_paths = {'Vegetables': "ho.pt", 'Fruit': "ho.pt", 'Product': "ho.pt", 'Fast Food': "fast.pt"}

# Glucose prediction data
coefficients = np.array([1.07142857, 0.25, 19.64285714, -0.58571429])
intercept = -209.99999999999216

data = {
    'HeartRate': [110, 115, 105, 120, 125],
    'SpO2': [98, 97, 96, 99, 98],
    'SkinTemp': [30, 32, 31, 29, 30],
    'IRValue': [1.2, 1.3, 1.1, 1.5, 1.4],
    'ActualGlucose': [110, 115, 105, 120, 125]
}

input_cases = [
    [120, 130, 125, 140, 135, 138, 142, 145, 147, 150, 152, 155,
     0.5, -0.3, 0.2, -0.1, 0.4, 0.3, -0.2, 0.1, 0.5, -0.4, 0.3, 0.2],
    [100, 102, 104, 106, 108, 110, 112, 115, 117, 118, 119, 120,
     -0.1, -0.2, 0.0, 0.1, 0.2, -0.1, -0.3, 0.2, 0.4, -0.2, 0.1, 0.0],
    [160, 158, 155, 150, 145, 140, 138, 135, 132, 130, 128, 125,
     0.3, 0.2, -0.1, -0.3, -0.4, 0.0, 0.1, 0.2, -0.2, 0.4, 0.3, -0.1]
]

current_index = 0
current_case = 0

# Food detection functions
def detect_objects(image, model_path):
    model = YOLO(model_path)
    prediction = model.predict(image)
    detected_labels = [int(box.cls[0]) for box in prediction[0].boxes]
    return detected_labels

def calculate_nutrition(detected_labels, nutrition_dict):
    total_carbs, total_calories = 0, 0
    detected_items = []
    for label in detected_labels:
        if label in nutrition_dict:
            item = nutrition_dict[label]
            total_carbs += item["carbs"]
            total_calories += item["calories"]
            detected_items.append(item)
    return total_carbs, total_calories, detected_items

# Glucose prediction functions
def load_model_and_scaler():
    model = load_model("model.h5", custom_objects={'mse': MeanSquaredError()})
    scaler = joblib.load("scaler.pkl")
    return model, scaler

def predict_glucose(hr, spo2, temp, ir):
    return intercept + (
        coefficients[0] * hr +
        coefficients[1] * spo2 +
        coefficients[2] * temp +
        coefficients[3] * ir
    )

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/food', methods=['GET', 'POST'])
def food_detection():
    if request.method == 'POST':
        category = request.form['category']
        file = request.files['file']
        if file:
            image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
            detected_labels = detect_objects(image, model_paths[category])
            total_carbs, total_calories, detected_items = calculate_nutrition(detected_labels, nutritional_info[category])
            return render_template('food_detection.html', detected_items=detected_items, total_carbs=total_carbs, total_calories=total_calories)
    return render_template('food_detection.html')

@app.route('/next')
def next_measurement():
    global current_index, current_case
    model, scaler = load_model_and_scaler()
    hr = data['HeartRate'][current_index]
    spo2 = data['SpO2'][current_index]
    temp = data['SkinTemp'][current_index]
    ir = data['IRValue'][current_index]
    actual_glucose = data['ActualGlucose'][current_index]
    predicted_now = predict_glucose(hr, spo2, temp, ir)
    input_data = input_cases[current_case]
    input_array = np.array([input_data], dtype=np.float32)
    input_scaled = scaler.transform(input_array).reshape(1, -1)
    predicted_later = model.predict(input_scaled)[0][0]
    if actual_glucose > 180:
        level = "danger"
    elif actual_glucose > 140:
        level = "warning"
    else:
        level = "normal"
    response = {
        "glucose": round(float(actual_glucose), 1),
        "predicted": round(float(predicted_now), 1),
        "predicted_later": round(float(predicted_later), 1),
        "level": level
    }
    current_index = (current_index + 1) % len(data['HeartRate'])
    current_case = (current_case + 1) % len(input_cases)
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)