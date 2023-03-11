import numpy as np 
from flask import Flask, request
from predict import make_prediction

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    """Basic HTML response."""
    body = (
        "<html>"
        "<body style='padding: 10px;'>"
        "<h1>Welcome to my Flask API</h1>"
        "</body>"
        "</html>"
    )
    return body

@app.route("/predict", methods=["POST"])
def predict():
    data_json = request.get_json()
    
    sepal_length = data_json["sepal_length"]
    sepal_width = data_json["sepal_width"]
    petal_length = data_json["petal_length"]
    petal_width = data_json["petal_width"]

    data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    predictions = make_prediction(data)
    
    return str(predictions)

if __name__ == "__main__":
    app.run()
