from joblib import load
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import cv2

from transformers import pipeline

#  import easyocr

import torch
from flask import Flask, request, jsonify
from PIL import Image


app = Flask(__name__)


carplate_haar_cascade = cv2.CascadeClassifier("haarcascade_russian_plate_number.xml")
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")


@app.route("/predict_images", methods=["POST"])
def process_image_1():
    for file in request.files:
        print(file)

    file = request.files["image"]
    # Read the image via file.stream
    #  img = Image.open(file.stream)

    img = Image.open(file.stream).convert("RGB")
    #  Neural network check
    #  model1 = load('new_model1.joblib')
    #  model2 = load('new_model2.joblib')
    model3 = load("new_model8.joblib")

    #  config1 = resolve_data_config({}, model=model1)
    #  config2 = resolve_data_config({}, model=model2)
    config3 = resolve_data_config({}, model=model3)

    #  transform1 = create_transform(**config1)
    #  transform2 = create_transform(**config2)
    transform3 = create_transform(**config3)

    #  tensor1 = transform1(img).unsqueeze(0)  # transform and add batch dimension
    #  tensor2 = transform2(img).unsqueeze(0)  # transform and add batch dimension
    tensor3 = transform3(img).unsqueeze(0)  # transform and add batch dimension

    #  out1 = model1(tensor1.to('cpu'))
    #  _, predicted1 = torch.max(out1, 1)

    # out2 = model2(tensor2.to('cpu'))
    # _, predicted2 = torch.max(out2, 1)

    out3 = model3(tensor3.to("cpu"))
    _, predicted3 = torch.max(out3, 1)
    return jsonify({"prediction": int(predicted3[0])})

    #  return jsonify({'msg': 'success', 'predictions': [int(predicted1[0]), int(predicted3[0]), int(predicted3[0])]})


@app.route("/predict_text", methods=["POST"])
def classify_transport_reviews():
    labels = ["artifacts", "animals", "food", "birds"]
    hypothesis_template = "This text is about {}."
    sequence = request.values["text"]

    prediction = classifier(sequence, labels, hypothesis_template=hypothesis_template, multi_labeled=True)

    return jsonify({"prediction": prediction["labels"][0]})


if __name__ == "__main__":
    app.run(debug=True)
