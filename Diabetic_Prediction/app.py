#import libraries
from flask import Flask,Request, jsonify
import joblib

# Initialize Flask app
app = Flask(__name__)

model = joblib.load('model.pickle')
scaler = joblib.load('scaler.pickle')

@app.route('/test', methods=['POST'])
def home():
    
      if Request.is_json:
        data = Request.get_json()
        features = [
            float(data['pregnancies']),
            float(data['glucose']),
            float(data['bloodPressure']),
            float(data['skinThickness']),
            float(data['insulin']),
            float(data['bmi']),
            float(data['diabetesPedigreeFunction']),
            float(data['age'])
        ]

        features = scaler.transform([features])
        prediction = model.predict(features)
        return jsonify(prediction=prediction.tolist())

if __name__ == '__main__':
    app.run(debug=True)