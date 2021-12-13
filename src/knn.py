import pandas as pd 
import numpy as np 
from sklearn.neighbors import KNeighborsClassifier
import preprocess
from sklearn.metrics import accuracy_score
import util


def KNN_baseline():
    """
    Implements the baseline model for K-nearest neighbors  
    """

    print("Training Baseline Model ... \n")
    dataset = preprocess.get_dataset()

    X_train,y_train = dataset.get("X_train"),dataset.get("y_train")
    X_test,y_test = dataset.get("X_test"),dataset.get("y_test")
    X_valid,y_valid = dataset.get("X_valid"),dataset.get("y_valid")

    #training the model
    knn = KNeighborsClassifier(n_neighbors=11)
    knn.fit(X_train, y_train)

    yhat = knn.predict(X_test)

    # predict on validation set
    val_pred = knn.predict(X_valid)

    # evaluate the model
    acc_train = knn.score(X_train, y_train)
    acc_val = accuracy_score(y_valid, val_pred)
    acc_test = accuracy_score(y_test, yhat)
    print('Accuracy on train set: %.2f' % (acc_train*100))
    print('Accuracy on test set: %.2f' % (acc_test*100))
    print('Accuracy on valid set: %.2f' % (acc_val*100))
    
    util.eval_mectrics(knn,X_test,y_test,save="CM_KNN_Baseline")
    

def KNN_final():
    """
    Implements the final model for k-nearesr neighbors  
    """

    print("Training Final Model ... \n")
    dataset = preprocess.get_dataset()

    X_train,y_train = dataset.get("X_train"),dataset.get("y_train")
    X_test,y_test = dataset.get("X_test"),dataset.get("y_test")
    X_valid,y_valid = dataset.get("X_valid"),dataset.get("y_valid")

    #training the model
    knn = KNeighborsClassifier(n_neighbors=2,weights = 'distance',metric='manhattan')
    knn.fit(X_train, y_train)

    yhat = knn.predict(X_test)

    # predict on validation set
    val_pred = knn.predict(X_valid)

    # evaluate the model
    acc_train = knn.score(X_train, y_train)
    acc_val = accuracy_score(y_valid, val_pred)
    acc_test = accuracy_score(y_test, yhat)
    print('Accuracy on train set: %.2f' % (acc_train*100))
    print('Accuracy on test set: %.2f' % (acc_test*100))
    print('Accuracy on valid set: %.2f' % (acc_val*100))
    
    util.eval_mectrics(knn,X_test,y_test,save="CM_KNN_Final")
