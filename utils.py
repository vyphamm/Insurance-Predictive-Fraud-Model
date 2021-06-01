from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import KFold
import sklearn as sk
from imblearn.ensemble import BalancedRandomForestClassifier 
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import numpy as np 
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import seaborn as sns

def oversample(X, y, percentage):
    over_sample = RandomOverSampler(sampling_strategy=percentage)
    return over_sample.fit_resample(X, y)

def compute_CV_recall(model, metric, X_train, Y_train, oversample_percent): 
    kf = KFold(n_splits = 4)
    validation_errors = []
    for train_idx, valid_idx in kf.split(X_train):
        # split the data
        split_X_train, split_X_valid = X_train.iloc[train_idx], X_train.iloc[valid_idx]
        split_Y_train, split_Y_valid = Y_train.iloc[train_idx], Y_train.iloc[valid_idx]
        # Fit the model on the training split
        
        split_X_train, split_Y_train = oversample(split_X_train, split_Y_train, oversample_percent)
        model.fit(split_X_train, split_Y_train.to_numpy())

        # Compute the RMSE on the validation split
        error = metric(split_Y_valid, model.predict(split_X_valid))
        validation_errors.append(error)
    return np.mean(validation_errors)

def display_report(model, X_train, Y_train, X_test, Y_test): 
    print("Training Accuracy: ", model.score(X_train, Y_train))
    print('Testing Accuracy: ', model.score(X_test, Y_test))
    cr = classification_report(Y_test,  model.predict(X_test))
    print(cr)
    
def display_ROC(Y_actual, Y_pred): 
    fpr, tpr, threshold = metrics.roc_curve(Y_actual, Y_pred)
    roc_auc = metrics.auc(fpr, tpr)
    f, ax = plt.subplots(figsize=(3, 3))
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()