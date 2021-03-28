from flask import Flask
from flask_restful import Api, Resource
from flask import render_template
from main import *
from urllib import parse as prs

app = Flask(__name__)
api = Api(app)

# render homepage
@app.route('/')
def home():
    return render_template('index.html')

# TODO take input from form, extract final parameter and pass to a redirect

# predict genre based on link parameter
class predict(Resource):
    def get(self, link):
        link = prs.quote(link)
        result = full_prediction(link)
        return {"data": result}

api.add_resource(predict, "/predict/<string:link>")


if __name__ == "__main__":
    app.run(debug=True)
    # https://www.youtube.com/watch?v=ZbZSe6N_BXs