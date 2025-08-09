#import libraries
from flask import Flask,request, jsonify

app = Flask(__name__)
@app.route('/', methods=['GET'])
def home():
    return jsonify(message="Welcome to the Diabetic Prediction App")

if __name__ == '__main__':
    app.run(debug=True)