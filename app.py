import numpy as np
import joblib
import tensorflow as tf

from flask import Flask, request, jsonify
from train import CoverTypeTrain


app = Flask(__name__)

load_models = {
    'knn': joblib.load('./models/knn.pkl'),
    'rf': joblib.load('./models/rf.pkl'),
    'nn': tf.keras.models.load_model('./models/nn.h5')
}

tree_types = {
    1.0: 'Spruce/Fir',
    2.0: 'Lodgepole Pine',
    3.0: 'Ponderosa Pine',
    4.0: 'Cottonwood/Willow',
    5.0: 'Aspen',
    6.0: 'Douglas-fir',
    7.0: 'Krummholz'
}

@app.route('/')
def predict():
    model_name = request.args.get('model')
    inputs = request.args.get('inputs')

    if model_name == 'heuristic':
        prediction = CoverTypeTrain.heuristic(inputs)
    elif model_name == 'knn':
        model = load_models[model_name]
        prediction = model.predict(inputs)
    elif model_name == 'nn':
        model = load_models[model_name]
        prediction = model.predict(inputs)
        prediction = np.argmax(prediction, axis=1)
    else:
        model = load_models[model_name]
        prediction = model.predict(inputs)

    output = {i: tree_types[i] for i in prediction}

    return jsonify(f'Prediction: {output}')
        