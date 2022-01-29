from flask import Flask,jsonify,request
from classifier import get_prediction

app=Flask(__name__)
@app.route("/")
def hello_world():
    return "hello everyone"

@app.route("/predict-alphabet",methods=["POST"])
def predict_data():
    image=request.files.get("alphabet")
    predict=get_prediction(image)
    return jsonify({
        "predict":predict
    }),200

if __name__ =="__main__":
    app.run(debug=True)
