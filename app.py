from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get values from form
    sepal_length = float(request.form['sepal_length'])
    sepal_width = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])
    
    # Convert to numpy array
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    # Predict
    prediction = model.predict(features)[0]
    species = ['Setosa', 'Versicolor', 'Virginica'][prediction]
    
    return render_template('index.html', prediction_text=f"Predicted species: {species}")

if __name__ == "__main__":
    app.run(debug=True)
