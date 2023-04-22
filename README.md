# About 

Training and evaluation for forest cover dataset with 4 diffrent models and simple API.

# Training and evaluation

To train provided model there is a class in train.py called CoverTypeTrain.

Example:

```python
from train import CoverTypeTrain

train = CoverTypeTrain()
train.load_data()
train.knn(neighbors=5, path_to_save='./models/knn_model.pkl')

```
To evaluate:

```python
from train import CoverTypeEvaluate

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




# Summary of models


* Heuristic

Simple heuristic method that based on elevation (first column).

Elevation is a feature that determines the height above sea level, it is given in meters.

I chose this trait after analyzing the plot below.

![alt text](./assets/elevation_by_cover_type.png)

I used a violin plot to visualize the relationship between labels and elevation.

We can see that most labels are in range of elevation that we can define.

For example label number 7 (Krummholz) is in the range 3200-3500.

However, we can also note that the range of label number 3 is almost the same as number 6.

This may be one of the main reasons why this model achieves low accuracy on the test set, as only 45%.


* K-Nearest Neighbors (KNN)

Algorithm performs very well with default arguments and without standardization.

Number of neighbors was set to 3 after analyzing plot below.

![alt text](./assets/knn_neighbors.png)

My pure intuition for choosing this model was that it makes sense that trees of the same type would be located near each other in a forest.

Model achieves around 98% accuracy on test set.


* Random Forest (RF) 

The best performing model, all arguments expect estimators are set to default.

Standardization is done with StandardScaler.

The number of estimators was set to 120 for best accuracy after analyzing plot below although 50 would be probably better for speed.

![alt text](./assets/rf_estimators.png)

Model achieves around 99% accuracy on test set.

A forest for a forest makes sense.

* Neural Network (NN)

Firstly, I found the best hyper-parametrs using keras-tuner library.

The description of that can be found in /models/hyperparametrs directory.

Then I trained net for 30 epochs on the training set, plots below show the process

![alt text](./assets/rf_estimators.png)

![alt text](./assets/rf_estimators.png)

Model achieves around 91% accuracy on test set.


# Evaluation

* Accuracy

![alt text](./assets/rf_estimators.png)

* Confusion matrix

![alt text](./assets/rf_estimators.png)

