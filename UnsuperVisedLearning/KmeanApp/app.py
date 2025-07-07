# Add jsonify to your imports
from flask import Flask, request, render_template, url_for, jsonify
import joblib
import numpy as np # Import numpy if you haven't already

app = Flask(__name__)

# --- Model Loading (no changes here) ---
scalar = joblib.load('scalar.lb')
KMn = joblib.load("crop_recommendation_model.lb")
df = joblib.load("Crop_recommendation_data.lb")

@app.route('/')
def home():
    return render_template('index.html')

# IMPORTANT: Ensure the route is '/predict'
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # --- Getting form data (no changes here) ---
            N = int(request.form['nitrogen'])
            P = int(request.form['phosphorus'])
            K = int(request.form['potassium'])
            temperature = float(request.form['temperature'])
            humidity = float(request.form['humidity'])
            ph = float(request.form['ph'])
            rainfall = float(request.form['rainfall'])

            # --- Prediction logic (no changes here) ---
            user_data = [[N, P, K, temperature, humidity, ph, rainfall]]
            trans_data = scalar.transform(user_data)
            prediction = KMn.predict(trans_data)[0]
            dt = dict(df[df['cluster_8'] == prediction]['label'].value_counts())

            # --- CHANGE THIS PART ---
            # Instead of rendering a template, return the dictionary as JSON
            return jsonify(dt)

        except Exception as e:
            # Return error as JSON with a 500 status code
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)