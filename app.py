from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import torch 
import torch.nn as nn
import os
from email_spam import ModelWrapper
vectorizer_path = 'vectorizer.pkl'

with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

with open(vectorizer_path, 'rb') as file:
    vectorizer = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/predict', methods=['POST'])
def predict():
    email_text = request.form['Text']
    email_vectorized = vectorizer.transform([email_text])
    
    prediction = model.predict(email_vectorized)
    output = 'Spam' if prediction[0] == 1 else 'Ham'

    return render_template('index1.html', prediction_text='Prediction: {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)