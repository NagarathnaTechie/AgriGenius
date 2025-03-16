from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np



# Initialize Flask app
app = Flask(__name__)

# Load the saved models and scalers
model = joblib.load('random_forest_model.pkl')
sc = joblib.load('standard_scaler.pkl')
ms = joblib.load('minmaxscaler.pkl')

# Define crop dictionary
df_dict = {
    1: 'Rice', 2: 'Maize', 3: 'Jute', 4: 'Cotton', 5: 'Coconut',
    6: 'Papaya', 7: 'Orange', 8: 'Apple', 9: 'Muskmelon', 10: 'Watermelon',
    11: 'Grapes', 12: 'Mango', 13: 'Banana', 14: 'Pomegranate', 15: 'Lentil',
    16: 'Blackgram', 17: 'Mungbean', 18: 'Mothbeans', 19: 'Pigeonpeas',
    20: 'Kidneybeans', 21: 'Chickpea', 22: 'Coffee'
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from form
    N = float(request.form['Nitrogen'])
    P = float(request.form['Phosphorus'])
    K = float(request.form['Potassium'])
    temperature = float(request.form['Temperature'])
    humidity = float(request.form['Humidity'])
    ph = float(request.form['Ph'])
    rainfall = float(request.form['Rainfall'])

    # Create a feature array
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

    # Apply transformations using the saved scalers
    transformed_features = ms.transform(features)
    transformed_features = sc.transform(transformed_features)

    # Make prediction
    prediction = model.predict(transformed_features)[0]

    # Map the predicted label to the crop name
    crop = df_dict.get(prediction, "No recommendation available")

    # Return the prediction as JSON for the pop-up
    return jsonify({'crop': crop})

if __name__ == '__main__':
    app.run(debug=True)
