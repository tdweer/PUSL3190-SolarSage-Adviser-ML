

import joblib
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask('ssaml')
CORS(app)

similarSolar = pd.read_csv('cssolar.csv')
cleanSolar = similarSolar.drop(columns=['ElectricityUsage', 'SystemSize', 'SystemType', 'SolarPanel',
                                        'Inverter', 'Battery ', 'NoOfBatteries ', 'NoOfInveters ', 'NoOfPanels'])

# Load the trained model
print('Opening pickle file...')
with open('trained_model.pkl', 'rb') as file:
    print('File opened successfully')
    model = joblib.load(file)
    print('Model loaded successfully')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()  # Get JSON data from POST request
        print('Received data:', data)  # Log the received data

        # Ensure that the feature order matches the model's training data
        input_features = ['SystemSizeID', 'SystemTypeID', 'SolarPanelID', 'BatteryID', 'InverterID']
        input_data = np.array([[data[feat] for feat in input_features]])  # Assuming input data is a dictionary

        # Use the trained model to predict
        predicted_cost = model.predict(input_data)
        formatted_predicted_cost = "{:.2f}".format(predicted_cost[0])

        print('=======================', predicted_cost)
        return jsonify({'predicted_cost': formatted_predicted_cost})
    except Exception as e:
        import traceback
        print('Error:', e)
        traceback.print_exc()  # Log the full traceback
        return jsonify({'error': str(e)}), 500
@app.route('/recommend', methods=['POST'])
def recommend_solar_system():
    try:
        data = request.json
        input_text = data.get('input_text')
        print(data)
        print(input_text)

        vc = CountVectorizer()
        cleanSolar['Tags'] = cleanSolar['Tags'].fillna('')
        tags = vc.fit_transform(cleanSolar['Tags'])
        user_input = vc.transform([input_text])
        similarities = cosine_similarity(user_input, tags)
        top_five = similarities.argsort()[0][::-1][:3]
        results = cleanSolar.loc[top_five, 'SolarSystem'].values.tolist()

        return jsonify({'recommendations': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
