import os 
import pandas as pd 
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

DATASET_PATH = "archive/Data"
def load_data(path=DATASET_PATH):
    csv_path = os.path.join(path,"features_3_sec.csv")
    return pd.read_csv(csv_path)

def get_labels():
    df = load_data()
    label = df['label'].unique()
    return label

def encoder(label):
    enc = OneHotEncoder(sparse=False)
    return enc.fit_transform(label)

def scaling(df):
    minmaxscalar = preprocessing.MinMaxScaler()
    return minmaxscalar.fit_transform(df)

def datasplit(x,y,firstsplit=0.30, secondsplit=0.50):
    X_train, X_test, y_train, y_test = train_test_split(x, y,test_size=firstsplit)
    print(X_test.shape)
    X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test,test_size=secondsplit)
    return {"X_train":X_train,"y_train":y_train,
            "X_test":X_test,"y_test":y_test,
            "X_valid":X_valid,"y_valid":y_valid}

def get_dataset():
    df = load_data()
    df = df.drop(['filename','length'], axis=1)
    x = df.loc[:, df.columns != 'label'] 
    y = df['label'].values.reshape(9990,1)

    #apply hot-ones encoding to the labels
    enc_y =  encoder(y) 
    #apply scaler to x values 
    x = scaling(x)
    
    return datasplit(x,enc_y)