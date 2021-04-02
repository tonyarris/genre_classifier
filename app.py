from flask import Flask
from flask_restful import Api, Resource, reqparse
from flask import render_template, request, jsonify
from main import full_prediction
import json

app = Flask(__name__)
api = Api(app)

@app.route('/analyse', methods=['GET'])
def analyse():
    link = request.args.get('link', type=str)
    result = full_prediction(link)
    return jsonify(result=result)

@app.route('/', methods=['GET'])
def home():
    if request.method == 'GET':
        return render_template('index.html')
    if request.method == 'POST':
        link = request.form.to_dict()
        result = full_prediction(link['link'])
        return jsonify({"data": result})

if __name__ == "__main__":
    app.run(debug=True)
