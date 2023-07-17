from joblib import load
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import cv2

import nltk

nltk.download("stopwords")
from nltk.corpus import stopwords
from string import punctuation

russian_stopwords = stopwords.words("russian")

import joblib
from keras.preprocessing.sequence import pad_sequences

from pymystem3 import Mystem

#  import easyocr
import numpy as np
import torch
from flask import Flask, request, jsonify
from PIL import Image

app = Flask(__name__)


carplate_haar_cascade = cv2.CascadeClassifier("haarcascade_russian_plate_number.xml")


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


def remove_punct(text):
    #  table = {33: '.', 34: ',', 35: '<', 36: '>', 37: '?', 38: '!', 39: '@', 40: '#', 41: '$', 42: '^', 43: '%', 44: '&', 45: '*', 46: '(', 47: ')', 58: '-', 59: '+', 60: '=', 61: '[', 62: ']', 63: '{', 64: '}', 91: ':', 92: ';', 93: '|', 94: '`', 95: '"', 96: '\'', 123: '/', 124: '~', 125: '№', 126: '\n'}
    #  return text.translate(table)
    table = {
        33: " ",
        34: " ",
        35: " ",
        36: " ",
        37: " ",
        38: " ",
        39: " ",
        40: " ",
        41: " ",
        42: " ",
        43: " ",
        44: " ",
        45: " ",
        46: " ",
        47: " ",
        58: " ",
        59: " ",
        60: " ",
        61: " ",
        62: " ",
        63: " ",
        64: " ",
        91: " ",
        92: " ",
        93: " ",
        94: " ",
        95: " ",
        96: " ",
        123: " ",
        124: " ",
        125: " ",
        126: " ",
    }
    return text.translate(table)


def preprocess_text(text, mystem, tokenizer, maxlen=100):
    text = text.lower()
    text = remove_punct(text)
    text = mystem.lemmatize(text)

    text = [
        token.strip()
        for token in text
        if token not in russian_stopwords and token != " " and token.strip() not in punctuation
    ]
    text = " ".join(text)

    final = tokenizer.texts_to_sequences([text])

    final = pad_sequences(final, padding="post", maxlen=maxlen)

    return final


@app.route("/predict_text", methods=["POST"])
def classify_transport_reviews():
    transport_classes = {0: "Безопасность", 1: "Комфорт"}

    mystem = Mystem()

    tokenizer = joblib.load("Tokenizer_transport.joblib")

    model = joblib.load("transport_classification1.joblib")

    review = request.data

    processed_review = preprocess_text(review, mystem, tokenizer)

    p = transport_classes[np.argmax(model.predict(processed_review))]

    return f"{p}"


if __name__ == "__main__":
    app.run(debug=True)
