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

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Dense(512, activation = tf.nn.elu,kernel_initializer="he_normal"))

    model.add(tf.keras.layers.Dense(256, activation = tf.nn.elu, kernel_initializer="he_normal"))

    model.add(tf.keras.layers.Dense(128, activation = tf.nn.elu, kernel_initializer="he_normal"))

    model.add(tf.keras.layers.Dense(64, activation = tf.nn.elu, kernel_initializer="he_normal"))


    model.add(tf.keras.layers.Dense(10, activation = tf.nn.softmax))

    model.compile(optimizer = 'adam', 
                loss = 'categorical_crossentropy',
                metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=40, validation_data = (X_valid,y_valid))

    model.evaluate(X_test,y_test)

    history = model.history 
    
    predictions = model.predict(X_test, batch_size=10, verbose=True)

    preds = np.argmax(predictions,axis=1)
    label = np.argmax(y_test,axis=1)

    cm = confusion_matrix(y_true=label,y_pred=preds)
    label = list(preprocess.get_labels())
    evaluate.plot_confusion_matrix(cm, label,title='Confusion matrix')

nn_model()





