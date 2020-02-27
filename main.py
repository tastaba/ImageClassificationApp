from flask import Flask, jsonify, request

app = Flask(__name__)
from model import model

cur_model = model()

@app.route("/image", methods=['POST'])
def do_prediction():
    f = request.files["image"]
    filepath = f.filename
    preprocess = cur_model.process(filepath)  # filename is passed because cv takes filepath as input.
    classification = cur_model.predict(preprocess)
    percent = cur_model.confidence(preprocess)
    return jsonify("Student is: " + classification + " with Confidence of: " + str(percent))

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)