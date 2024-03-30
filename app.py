import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
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
        data = request.get_json()  # Get JSON data from POST request
        print(data)
        data = [data]
        # Convert JSON data to DataFrame
        input_data = pd.DataFrame(data)

        # Use the trained model to predict
        predicted_cost = model.predict(input_data)
        formatted_predicted_cost = "{:.2f}".format(predicted_cost[0])

        print('=======================', predicted_cost)
        return jsonify({'predicted_cost': formatted_predicted_cost})


    @app.route('/recommend', methods=['POST'])
    def recommend_solar_system():
        data = request.json
        input_text = data.get('input_text')

        vc = CountVectorizer()
        cleanSolar['Tags'] = cleanSolar['Tags'].fillna('')
        tags = vc.fit_transform(cleanSolar['Tags'])
        user_input = vc.transform([input_text])
        similarities = cosine_similarity(user_input, tags)
        top_five = similarities.argsort()[0][::-1][:5]
        results = cleanSolar.loc[top_five, 'SolarSystem'].values.tolist()

        return jsonify({'recommendations': results})


    if __name__ == '__main__':
        app.run(debug=True)