from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib

app = Flask('ssaml')
CORS(app)

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

if __name__ == '__main__':
    app.run(debug=True)
