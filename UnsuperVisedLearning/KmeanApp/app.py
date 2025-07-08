from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np
import traceback

app = Flask(__name__)

try:
    scalar = joblib.load('scalar.lb')
    KMn = joblib.load("crop_recommendation_model.lb")
    df = joblib.load("Crop_recommendation_data.lb")
except Exception as e:
    print("Model/Data loading failed:", e)
    traceback.print_exc()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorus'])
        K = int(request.form['potassium'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        user_data = [[N, P, K, temperature, humidity, ph, rainfall]]
        trans_data = scalar.transform(user_data)
        prediction = KMn.predict(trans_data)[0]
        dt = dict(df[df['cluster_8'] == prediction]['label'].value_counts())

        return jsonify(dt)

    except Exception as e:
        print("Prediction failed:", e)
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
