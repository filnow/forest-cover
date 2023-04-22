import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf
import keras_tuner as kt
import seaborn as sns

from typing import List
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


class ParamsSearch:
    '''
    Class for hyperparameters search

    Args:
        X_train (np.ndarray): train data
        y_train (np.ndarray): train labels
        X_val (np.ndarray): validation data
        y_val (np.ndarray): validation labels
        num_classes (int): number of classes
    
    Methods:
        build(hp: kt.HyperParameters): build model
        hyper_search(): search hyperparameters

    '''
    def __init__(self, 
                 X_train: np.ndarray, 
                 y_train: np.ndarray, 
                 X_val: np.ndarray, 
                 y_val: np.ndarray, 
                 num_classes: int) -> None:
        
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.num_classes = num_classes
        self.tuner = None

    def build(self, hp: kt.HyperParameters) -> tf.keras.Sequential:
        '''
        Build model

        Default parameters:
            num_layers: 3
            units: 128
            activation: relu
            dropout: False
            lr: 0.0002
        Args:
            hp (kt.HyperParameters): hyperparameters
        Returns:
            model (tf.keras.Sequential): model
        
        '''
        model = tf.keras.Sequential()
        for i in range(hp.Int("num_layers", 3, 6, default=3)):
            model.add(
                tf.keras.layers.Dense(
                    units=hp.Int(f"units_{i}", min_value=32, max_value=512, step=32, default=128),
                    activation=hp.Choice("activation", ["relu", "elu"], default="relu"),
                )
            )
            model.add(tf.keras.layers.BatchNormalization())
        if hp.Boolean("dropout", default=False):
            model.add(tf.keras.layers.Dropout(rate=0.25))
        model.add(tf.keras.layers.Dense(self.num_classes, activation="softmax"))
        learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log", default=0.0002)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model
    
    def hyper_search(self) -> None:
        '''
        Search hyperparameters

        Max trials: 20 - number of models to train
        Executions per trial: 2 - number of times to train each model
        Epochs: 10 - number of epochs to train each model

        '''
        self.tuner = kt.RandomSearch(
            hypermodel=self.build,
            objective="val_accuracy",
            max_trials=10,
            executions_per_trial=2,
            overwrite=True,
            directory="./models",
            project_name="hyperparametrs_search",
        )

        self.tuner.search(self.X_train, self.y_train, epochs=10, validation_data=(self.X_val, self.y_val))


class CoverTypeTrain:
    '''
    Class for training models

    Methods:
        load_data(): load data
        knn(): train KNN model
        rf(): train RF model
        nn(): train NN model
        heuristic(): heuristic model

    '''
    def __init__(self) -> None:
        self.path = './data/covtype.data'
        self.data = None
        self.X_test, self.y_test = None, None

    def load_data(self) -> None:
        '''
        Load data, change one-hot encoding to categorical,
        drop one-hot encoded columns,
        split data to train, validation and test

        '''
        self.data = pd.read_csv(self.path, header=None, sep=',')
        self.y = self.data.iloc[:, -1].values

        wilderness_areas = self.data.iloc[:, 10:14]
        wilderness_areas.columns = list(range(1, 5))
        self.data['10'] = wilderness_areas.idxmax(axis=1)

        soil_types = self.data.iloc[:, 14:54]
        soil_types.columns = list(range(1, 41))
        self.data['11'] = soil_types.idxmax(axis=1)
                
        self.data = self.data.drop(columns=self.data.columns[10:55])

        self.X = self.data.values

        #NOTE: split data to train and test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        #NOTE: split train data to train and validation
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train, test_size=0.1, random_state=42)

    def knn(self, 
            neighbors: int = 3, 
            path_to_save: str = './models/knn.pkl') -> None:
        '''
        Method for training KNN model without scaling

        Args:
            neighbors (int): number of neighbors
            path_to_save (str): path to save model

        '''
        model = Pipeline([
            ('knn', KNeighborsClassifier(n_neighbors=neighbors))
        ])
        model.fit(self.X_train, self.y_train)
        joblib.dump(model, path_to_save)

        print(f'Model trained and saved to {path_to_save}')

    def random_forest(self, 
                      n_estimators: int = 100, 
                      path_to_save: str = './models/rf.pkl') -> None:
        '''
        Method for training Random Forest model with scaling
        
        Args:
            n_estimators (int): number of estimators
            path_to_save (str): path to save model
        
        '''
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('random_forest', RandomForestClassifier(n_estimators=n_estimators))
        ])
        model.fit(self.X_train, self.y_train)
        joblib.dump(model, path_to_save)

        print(f'Model trained and saved to {path_to_save}')

    def nn(self, 
           path_to_save: str = './models/nn.h5', 
           search: bool = False, 
           plots: bool = False,
           epochs: int = 30) -> None:
        '''
        Method for training NN model with scaling

        Args:
            path_to_save (str): path to save model
            search (bool): search hyperparameters
            plots (bool): show plots
            epochs (int): number of epochs

        '''
        y_train_one_hot = tf.keras.utils.to_categorical(self.y_train)
        y_val_one_hot = tf.keras.utils.to_categorical(self.y_val)

        #NOTE: search on fraction of data for faster results
        params = ParamsSearch(self.X_train[:100000], 
                              y_train_one_hot[:100000], 
                              self.X_val[:10000], 
                              y_val_one_hot[:10000], 
                              max(self.y_train)+1)
        if search:
            params.hyper_search()
            best_hps = params.tuner.get_best_hyperparameters(5)
            model = params.build(best_hps[0])
        else:
            #NOTE: pass a empty hyperparameters object to build the model with default hyperparameters
            model = params.build(kt.HyperParameters())

        model.fit(self.X_train, y_train_one_hot, epochs=epochs, validation_data=(self.X_val, y_val_one_hot))

        if plots:
            plt.figure()
            plt.plot(model.history.history['accuracy'])
            plt.plot(model.history.history['val_accuracy'])
            plt.title('Model accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend(['train', 'val'], loc='upper left')
            plt.savefig('./assets/accuracy_of_nn.png', facecolor='white')

            plt.figure()
            plt.plot(model.history.history['loss'])
            plt.plot(model.history.history['val_loss'])
            plt.title('Model loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['train', 'val'], loc='upper left')
            plt.savefig('./assets/loss_of_nn.png', facecolor='white')

        model.save(path_to_save)

    @staticmethod
    def heuristic(elevation_list: List[int]) -> List[float]:
        '''
        Heuristic model for predicting cover type based on elevation

        Args:
            elevation_list (List): list of elevations
        Returns:
            List: list of cover types

        '''
        elevation_ranges = {
            1.0: (3100, 3300),
            2.0: (2900, 3100),
            3.0: (2500, 2700),
            4.0: (1500, 2300),
            5.0: (2700, 2900),
            6.0: (2300, 2500),
            7.0: (3300, 4000)
        }
        def predict(elevation: int) -> float:
            for cover_type, elevation_range in elevation_ranges.items():
                if elevation >= elevation_range[0] and elevation <= elevation_range[1]:
                    return cover_type
                elif elevation < 1500:
                    return 4.0
                elif elevation > 4000:
                    return 7.0
            raise ValueError('Elevation out of range')
        
        return [predict(elevation) for elevation in elevation_list]
    

class CoverTypeEvaluate:
    '''
    Class for evaluating models
    
    Args:
        X_test (np.array): test data
        y_test (np.array): test labels
    
    '''
    def __init__(self, 
                 X_test: np.ndarray, 
                 y_test: np.ndarray,
                 knn_path: str = './models/knn.pkl',
                 rf_path: str = './models/rf.pkl',
                 nn_path: str = './models/nn.h5') -> None:
        self.X_test = X_test
        self.y_test = y_test

        self.knn = joblib.load(knn_path)
        self.rf = joblib.load(rf_path)
        self.nn = tf.keras.models.load_model(nn_path)
        self.heuristic = CoverTypeTrain.heuristic

    def metrics(self, model_name: str, metric: str) -> float:
        '''
        Method for evaluating models

        Args:
            model_name (str): name of model
            metric (str): metric to evaluate
        Returns:
            float: value of metric

        '''
        model = getattr(self, model_name)

        if model_name == 'nn':
            y_pred = np.argmax(model.predict(self.X_test), axis=1)
        elif model_name == 'heuristic':
            y_pred = self.heuristic(self.X_test[:, 0])
        else:
            y_pred = model.predict(self.X_test)

        if metric == 'accuracy':
            return accuracy_score(self.y_test, y_pred)
        elif metric == 'precision':
            return precision_score(self.y_test, y_pred, average='weighted')
        elif metric == 'recall':
            return recall_score(self.y_test, y_pred, average='weighted')
        elif metric == 'f1':
            return f1_score(self.y_test, y_pred, average='weighted')
        elif metric == 'confusion_matrix':
            return confusion_matrix(self.y_test, y_pred)
    
    def plot_accuracies(self) -> None:
        '''
        Method for plotting accuracies of different models

        '''
        accuracies = {
            'KNN': self.metrics(model_name='knn', metric='accuracy'),
            'Random Forest': self.metrics(model_name='rf', metric='accuracy'),
            'Neural Network': self.metrics(model_name='nn', metric='accuracy'),
            'Heuristic': self.metrics(model_name='heuristic', metric='accuracy')
        }

        plt.figure(figsize=(8,6))
        plt.bar(accuracies.keys(), accuracies.values(), color='blue')
        plt.title('Accuracy Scores of Different Models')
        plt.xlabel('Model')
        plt.ylabel('Accuracy')
        plt.ylim([0, 1.1])

        for i, v in enumerate(accuracies.values()):
            plt.text(i, v+0.01, str(round(v,2)), ha='center')

        plt.savefig('./assets/accuracies.png', facecolor='white')

    def plot_confusion_matrices(self) -> None:
        '''
        Method for plotting confusion matrices of different models

        '''
        confusion_matrices = {
            'KNN': self.metrics(model_name='knn', metric='confusion_matrix'),
            'Random Forest': self.metrics(model_name='rf', metric='confusion_matrix'),
            'Neural Network': self.metrics(model_name='nn', metric='confusion_matrix'),
            'Heuristic': self.metrics(model_name='heuristic', metric='confusion_matrix')
        }

        fig, axs = plt.subplots(2, 2, figsize=(12, 12))
        fig.suptitle('Confusion Matrices of Different Models')
        
        for i, (model_name, confusion_matrix) in enumerate(confusion_matrices.items()):
            ax = axs[i//2, i%2]
            ax.set_title(model_name)
            sns.heatmap(confusion_matrix, annot=True, ax=ax, fmt='d')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
        
        plt.savefig('./assets/confusion_matrices.png', facecolor='white')
