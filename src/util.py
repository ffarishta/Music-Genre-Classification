from sklearn.metrics import confusion_matrix
import numpy as np
import itertools
import matplotlib.pyplot as plt
import os 
import preprocess


def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues, save= "hello"):

    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize=(12, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    os.chdir("../plots")
    path =  os.path.abspath(os.curdir)
    img_path = os.path.join(path,save)
    plt.savefig(img_path)

    plt.clf()


def eval_mectrics(model,X_test,y_test,save=" ",history = None,argmax=True):
    """
    Applies evaluation to the models   
    """
    
    predictions = model.predict(X_test)

    if argmax:
        preds = np.argmax(predictions,axis=1)
        label = np.argmax(y_test,axis=1)
    else:
        preds = predictions
        label = y_test

    cm = confusion_matrix(y_true=label,y_pred=preds)
    label = list(preprocess.get_labels())
    plot_confusion_matrix(cm, label,save=save,title='Confusion matrix')

def plot_acc_err(hist,save):
    plt.figure(figsize=(12, 6))
    plt.figure(figsize=(20,15))
    fig, axs = plt.subplots(2)
    axs[0].plot(hist.history["accuracy"], label="train accuracy")
    axs[0].plot(hist.history["val_accuracy"], label="Dev accuracy")    
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")
    # Error 
    axs[1].plot(hist.history["loss"], label="train error")
    axs[1].plot(hist.history["val_loss"], label="Dev error")    
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")


    path =  os.path.abspath(os.curdir)
    img_path = os.path.join(path,save)
    plt.savefig(img_path)

