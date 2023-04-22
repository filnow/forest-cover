import numpy as np
import joblib
import tensorflow as tf
import json

from flask import Flask, request, jsonify
from utils import CoverTypeTrain


app = Flask(__name__)

load_models = {
    'knn': joblib.load('./models/knn.pkl'),
    'rf': joblib.load('./models/rf.pkl'),
    'nn': tf.keras.models.load_model('./models/nn.h5'),
}

#NOTE: tree types from data/convtype.info
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

    #NOTE: check for improper inputs
    try:
        inputs_dict = json.loads(inputs)
        required_keys = ['Elevation', 
                         'Aspect', 
                         'Slope', 
                         'Horizontal_Distance_To_Hydrology', 
                         'Vertical_Distance_To_Hydrology',
                         'Horizontal_Distance_To_Roadways',
                         'Hillshade_9am',
                         'Hillshade_Noon',
                         'Hillshade_3pm',
                         'Horizontal_Distance_To_Fire_Points',
                         'Wilderness_Area',
                         'Soil_Type']
        for key in required_keys:
            if key not in inputs_dict:
                return jsonify(f"Missing required key: {key}")
            if not isinstance(inputs_dict[key], (float, int)):
                return jsonify(f"{key} must be a numeric value")
    except json.JSONDecodeError:
        return jsonify('Invalid input data')

    if model_name == 'heuristic':
        prediction = CoverTypeTrain.heuristic([inputs_dict.get('Elevation')])
    elif model_name == 'knn':
        model = load_models[model_name]
        prediction = model.predict([[*inputs_dict.values()]])
    elif model_name == 'nn':
        model = load_models[model_name]
        prediction = model.predict([[*inputs_dict.values()]])
        prediction = np.argmax(prediction, axis=1)
    else:
        model = load_models[model_name]
        prediction = model.predict([[*inputs_dict.values()]])

    output = {i: tree_types[i] for i in prediction}

    return jsonify(f'{output}')
        

if __name__ == '__main__':
    app.run(debug=True)