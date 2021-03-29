from flask import Flask
from flask_restful import Api, Resource, reqparse
from flask import render_template, request
from main import full_prediction

app = Flask(__name__)
api = Api(app)

# render homepage
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    link =request.form['link']
    result = full_prediction(link)
    return {"data": result}, 200

if __name__ == "__main__":
    app.run(debug=True)
