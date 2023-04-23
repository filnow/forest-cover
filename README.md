# About 

Training and evaluation for forest cover dataset with 4 diffrent models and simple API.

# Training and evaluation

To train provided model there is a class in utils.py called CoverTypeTrain.

Example:

```python
from utils import CoverTypeTrain

train = CoverTypeTrain()
train.load_data()
train.knn(neighbors=5, path_to_save='./models/knn_model.pkl')

```
To evaluate:

```python
from utils import CoverTypeEvaluate

evaluate = CoverTypeEvaluate(X_test=train.X_test,
                             y_test=train.y_test,
                             knn_path='./models/knn_model.pkl')
accuracy = evaluate.metrics(model_name='knn', metric='accuracy')
print(accuracy)

```

# Run API

```bash

git clone https://github.com/filnow/forest-cover.git
cd forest-cover
pip install -r requirements.txt
python3 app.py

```
## Docker run

```bash
docker build -t forest-cover .
docker run -p 5000:5000 forest-cover

```

## Example request in python

Data comes from third row of the dataset that is labeled as 2 - Lodgepole Pine.

```python
import requests
import json

url = 'http://localhost:5000/' 

input_data = {
    'Elevation': [2785],
    'Aspect': [155],
    'Slope': [18],
    'Horizontal_Distance_To_Hydrology': [242],
    'Vertical_Distance_To_Hydrology': [118],
    'Horizontal_Distance_To_Roadways': [3090],
    'Hillshade_9am': [238],
    'Hillshade_Noon': [238],
    'Hillshade_3pm': [122],
    'Horizontal_Distance_To_Fire_Points': [6211],
    'Wilderness_Area': [1],
    'Soil_Type': [30]
}

input_data_json = json.dumps(input_data)

response = requests.get(url, params={'model': 'knn', 'inputs': input_data_json})

print(response.text)
```
For multiple inputs just add values to arrays.

## Supported models names

* nn - Neural Network

* knn - K-Nearest Neighbors 

* rf - Random Forest

* heuristic - Heuristic based on elevation

# Summary of models


## Heuristic

Simple heuristic method that based on elevation (first column).
Elevation is a feature that determines the height above sea level, it is given in meters.
I chose this trait after analyzing the plot below.

![alt text](./assets/elevation_by_cover_type.png)

I used a violin plot to visualize the relationship between labels and elevation.
We can see that most labels are in range of elevation that we can define.
For example label number 7 (Krummholz) is in the range 3200-3500.
However, we can also note that the range of label number 3 is almost the same as number 6.
This may be one of the main reasons why this model achieves low accuracy on the test set, as only 45%.


## K-Nearest Neighbors (KNN)

Algorithm performs very well with default arguments and without standardization.
Number of neighbors was set to 3 after analyzing plot below.

![alt text](./assets/knn_neighbors.png)

My pure intuition for choosing this model was that it makes sense that trees of the same type would be located near each other in a forest.

Model achieves around 97% accuracy on test set.


## Random Forest (RF) 

All arguments for this algorithm expect estimators are set to default.
Standardization is done with StandardScaler.
The number of estimators was set to 120 for best accuracy after analyzing plot below although 50 would be probably better for speed.

![alt text](./assets/rf_estimators.png)

Model achieves around 96% accuracy on test set.

A forest for a forest makes sense.


## Neural Network (NN)

Firstly, I found the best hyper-parametrs using keras-tuner library.
The description of that can be found in /models/hyperparametrs directory.
Then I trained net for 30 epochs on the training set, plots below show the process

![alt text](./assets/accuracy_of_nn.png)

![alt text](./assets/loss_of_nn.png)

Model achieves around 91% accuracy on test set.


# Evaluation

## Accuracy

![alt text](./assets/accuracies.png)

## Confusion matrix

![alt text](./assets/confusion_matrices.png)


# Additional infomations

Model rl.pkl is random forest model trained on 10 estimators beacause of github big files limitations

You can easily train another model with 120 estimators on your own using CoverTypeTrain.
