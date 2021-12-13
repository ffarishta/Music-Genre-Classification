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
import tensorflow as tf
from sklearn.metrics import classification_report

def baseline_nn_model():
    """
    Implements the baseline model for neural networks  
    """

    print("Training Baseline Model ... \n")

    dataset = preprocess.get_dataset()

    X_train,y_train = dataset.get("X_train"),dataset.get("y_train")
    X_test,y_test = dataset.get("X_test"),dataset.get("y_test")
    X_valid,y_valid = dataset.get("X_valid"),dataset.get("y_valid")

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(512, activation = tf.nn.relu))
    model.add(tf.keras.layers.Dense(256, activation = tf.nn.relu))
    model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
    model.add(tf.keras.layers.Dense(64, activation = tf.nn.relu))
    model.add(tf.keras.layers.Dense(16, activation = tf.nn.relu))
    model.add(tf.keras.layers.Dense(10, activation = tf.nn.softmax))

    model.compile(optimizer = 'adam', 
                loss = 'categorical_crossentropy',
                metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=50, validation_data = (X_valid,y_valid))

    print("Testing Baseline Model ... \n")

    history = model.history 

    #util.plot_acc_err(history,save="unregularized")
    
    predictions = model.predict(X_test, batch_size=10, verbose=True)

    preds = np.argmax(predictions,axis=1)
    label = np.argmax(y_test,axis=1)

    p1 = preds
    l1 = label

    cm = confusion_matrix(y_true=label,y_pred=preds)
    label = list(preprocess.get_labels())
    util.plot_confusion_matrix(cm, label,title='Confusion matrix', save = "CM_NN_Baseline.png")

    print(classification_report(l1,p1))

#baseline_nn_model()





