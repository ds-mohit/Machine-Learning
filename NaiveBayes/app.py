from flask import Flask, render_template, request, url_for
import pandas as pd
import joblib,pickle
import sqlite3

vectorizer = pickle.load(open('vectorizer.lb',"rb"))
model = pickle.load(open("spam_classifier.pkl", "rb"))

app = Flask(__name__)

@app.route('/',methods=['GET','POST'])
def home():
    return render_template("index.html")

@app.route('/predict',methods=['POST'])
def predict():
    try:
        if request.method == 'POST':
            msg = str(request.form['msg'])
            transformed = vectorizer.transform([msg])
            transformed_data = vectorizer.toarray()
            pred = model.predict(transformed_data)
            output = str(pred[0])
            return render_template('result.html',prediction = f"{output}")
    except Exception as e:
        return render_template('result.html',prediction = f"Error :{str(e)}")

    return render_template("index.html")



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)