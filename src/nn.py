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
import util
import itertools
from sklearn.metrics import classification_report

def nn_model():
    """
    Implements the final model for neural networks  
    """

    print("Training Final Model ... \n")

    dataset = preprocess.get_dataset()

    X_train,y_train = dataset.get("X_train"),dataset.get("y_train")
    X_test,y_test = dataset.get("X_test"),dataset.get("y_test")
    X_valid,y_valid = dataset.get("X_valid"),dataset.get("y_valid")

    model = tf.keras.Sequential()

    model = keras.Sequential()

    model.add(keras.layers.Dense(512, activation = "relu",kernel_initializer="he_normal"))

    model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.Dense(256, activation = "relu", kernel_initializer="he_normal"))

    model.add(keras.layers.Dropout(0.1))

    model.add(keras.layers.Dense(128, activation = "relu", kernel_initializer="he_normal"))

    model.add(keras.layers.Dropout(0.1))

    model.add(keras.layers.Dense(64, activation = "relu", kernel_initializer="he_normal"))


    model.add(keras.layers.Dense(10, activation = "softmax"))

    model.compile(optimizer = 'adam', 
                loss = 'categorical_crossentropy',
                metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=60, validation_data = (X_valid,y_valid))

    print("Testing Final Model ... \n")

    model.evaluate(X_test,y_test)

    history = model.history 
    
    predictions = model.predict(X_test, batch_size=10, verbose=True)

    preds = np.argmax(predictions,axis=1)
    label = np.argmax(y_test,axis=1)

    p1 = preds
    l1 = label

    
    cm = confusion_matrix(y_true=label,y_pred=preds)
    label = list(preprocess.get_labels())
    util.plot_confusion_matrix(cm, label,title='Confusion matrix', save = "CM_NN_Dropout")

    print(classification_report(l1,p1))
    

#nn_model()





