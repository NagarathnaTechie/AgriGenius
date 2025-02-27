from flask import Flask, request, render_template
import numpy as np
import joblib

# Load the model and scalers using joblib
model = joblib.load('random_forest_model.pkl')
sc = joblib.load('standard_scaler.pkl')
ms = joblib.load('minmaxscaler.pkl')

# Create Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    # Get form data
    N = request.form['Nitrogen']
    P = request.form['Phosphorus']
    K = request.form['Potassium']
    temp = request.form['Temperature']
    humidity = request.form['Humidity']
    ph = request.form['Ph']
    rainfall = request.form['Rainfall']

    # Prepare feature list and transform
    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list, dtype=float).reshape(1, -1)

    # Scale and standardize features
    scaled_features = ms.transform(single_pred)
    final_features = sc.transform(scaled_features)

    # Predict using the model
    prediction = model.predict(final_features)

    # Crop dictionary for mapping
    crop_dict = {
        1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut",
        6: "Papaya", 7: "Orange", 8: "Apple", 9: "Muskmelon", 10: "Watermelon",
        11: "Grapes", 12: "Mango", 13: "Banana", 14: "Pomegranate", 15: "Lentil",
        16: "Blackgram", 17: "Mungbean", 18: "Mothbeans", 19: "Pigeonpeas",
        20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
    }

    # Get the prediction result
    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        result = "{} is the best crop to be cultivated right there".format(crop)
    else:
        result = "Sorry, we could not determine the best crop to be cultivated with the provided data."
    
    return render_template('index.html', result=result)

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
