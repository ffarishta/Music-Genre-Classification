import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from keras.callbacks import CSVLogger
from sklearn import metrics 
import preprocess
from sklearn.metrics import confusion_matrix
import evaluate
import itertools

def nn_model():

    dataset = preprocess.get_dataset()

    X_train,y_train = dataset.get("X_train"),dataset.get("y_train")
    X_test,y_test = dataset.get("X_train"),dataset.get("y_train")
    X_valid,y_valid = dataset.get("X_train"),dataset.get("y_train")

    model = keras.Sequential()

    #model.add(keras.layers.Dense(1024, activation = "relu"))

    model.add(keras.layers.Dense(512, activation = "sigmoid"))
    
    model.add(keras.layers.Dense(256, activation = "sigmoid"))
    
    model.add(keras.layers.Dense(128, activation = "sigmoid"))
    
    model.add(keras.layers.Dense(64, activation = "sigmoid"))
    
    model.add(keras.layers.Dense(10, activation = "sigmoid"))

    model.compile(optimizer = 'adam', 
                loss = 'categorical_crossentropy',
                metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=40, validation_data = (X_valid,y_valid))

nn_model()





