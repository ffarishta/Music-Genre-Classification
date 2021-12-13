import pandas as pd 
import numpy as np 
from sklearn.linear_model import LogisticRegression
import preprocess
from sklearn.metrics import accuracy_score
import util

def LR_baseline():
    """
    Implements the baseline model for logistic regression 
    """

    print("Training Baseline Model ... \n")

    #importing dataset
    dataset = preprocess.get_dataset(labelenc=True)

    X_train,y_train = dataset.get("X_train"),dataset.get("y_train")
    X_test,y_test = dataset.get("X_test"),dataset.get("y_test")
    X_valid,y_valid = dataset.get("X_valid"),dataset.get("y_valid")

    #training the model
    model = LogisticRegression(multi_class='multinomial',solver='lbfgs', max_iter=1000)
    model.fit(X_train, np.ravel(y_train,order='C'))
    
    yhat = model.predict(X_test)

    # predict on validation set
    val_pred = model.predict(X_valid)

    # evaluate the model
    acc_train = model.score(X_train, y_train)
    acc_val = accuracy_score(y_valid, val_pred)
    acc_test = accuracy_score(y_test, yhat)
    print('Accuracy on train set: %.2f' % (acc_train*100))
    print('Accuracy on test set: %.2f' % (acc_test*100))
    print('Accuracy on valid set: %.2f' % (acc_val*100))

    util.eval_mectrics(model,X_test,y_test,save="CM_LR_Baseline",argmax=False)

    
#LR_baseline()
def LR_final():
    """
    Implements the final model for logistic regression  
    """

    print("Training Final Model ... \n")

    # importing dataset 
    dataset = preprocess.get_dataset(labelenc=True)

    X_train,y_train = dataset.get("X_train"),dataset.get("y_train")
    X_test,y_test = dataset.get("X_test"),dataset.get("y_test")
    X_valid,y_valid = dataset.get("X_valid"),dataset.get("y_valid")

    #training the model 
    model = LogisticRegression(multi_class='multinomial',solver='lbfgs', max_iter=1000, C = 15, class_weight = "balanced")
    model.fit(X_train, np.ravel(y_train,order='C'))
    
    yhat = model.predict(X_test)

    # predict on validation set
    val_pred = model.predict(X_valid)

    # evaluate the model
    acc_train = model.score(X_train, y_train)
    acc_val = accuracy_score(y_valid, val_pred)
    acc_test = accuracy_score(y_test, yhat)
    print('Accuracy on train set: %.2f' % (acc_train*100))
    print('Accuracy on test set: %.2f' % (acc_test*100))
    print('Accuracy on valid set: %.2f' % (acc_val*100))

    util.eval_mectrics(model,X_test,y_test,save="CM_LR_Final",argmax=False)

#LR_final()