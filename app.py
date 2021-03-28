from flask import Flask
from flask_restful import Api, Resource
from main import *
from urllib import parse as prs

app = Flask(__name__)
api = Api(app)

class predict(Resource):
    def get(self, link):
        link = prs.quote(link)
        result = full_prediction(link)
        return {"data": result}

api.add_resource(predict, "/predict/<string:link>")

if __name__ == "__main__":
    app.run(debug=True)
    # https://www.youtube.com/watch?v=ZbZSe6N_BXs