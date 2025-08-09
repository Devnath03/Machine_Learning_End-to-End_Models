#import libraries
from flask import Flask, request, jsonify
import joblib

# Initialize Flask app
app = Flask(__name__)

model = joblib.load('model.pickle')
scaler = joblib.load('scaler.pickle')

@app.route('/test', methods=['POST'])
def home():
    try:
      if request.is_json:
        data = request.get_json()
        print(data)

        features = [
    float(data['pregnancies']),
    float(data['glucose']),
    float(data['bloodpressure']),
    float(data['skinthickness']),
    float(data['insulin']),
    float(data['bmi']),
    float(data['diabetespedigree']),
    float(data['age'])
    ]

        print(features)

        features = scaler.transform([features])
        prediction = model.predict(features)

        result = "Diabetics" if prediction[0] == 1 else "Non-Diabetics"
        return jsonify({
            "prediction": result
        })

    except Exception as e:
        return jsonify({
            "prediction": "error"
        }), 400

if __name__ == '__main__':
    app.run(debug=True)