#import libraries
from flask import Flask,request, jsonify
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
        features = scaler.transform([data['features']])
        prediction = model.predict(features)
        return jsonify(prediction=prediction.tolist())
    except Exception as e:
        return jsonify(error=str(e)), 400

if __name__ == '__main__':
    app.run(debug=True)