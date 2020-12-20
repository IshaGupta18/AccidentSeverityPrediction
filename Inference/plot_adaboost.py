#required librarys
import numpy as np 
import sys
#For analysing model
import joblib
from sklearn.metrics import mean_squared_log_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import zero_one_loss

if __name__ == "__main__":
    loaded_model = joblib.load(f"../../storage/models/{sys.argv[1]}.joblib")
    ada_discrete = loaded_model.best_estimator_
    print("model loaded")
    X_train = pd.read_csv("../../storage/data/processed/traning_data.csv")
    X_test = pd.read_csv("../../storage/data/processed/testing_data.csv")
    y_train = np.load("../../storage/data/processed/y_train.npy")
    y_test = np.load("../../storage/data/processed/y_test.npy")
    print("data loaded")
    
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    n_estimators = 250
    ada_discrete_err = np.zeros((n_estimators,))
    for i, y_pred in enumerate(ada_discrete.staged_predict(X_test)):
        ada_discrete_err[i] = zero_one_loss(y_pred, y_test)
    ada_discrete_err_train = np.zeros((n_estimators,))
    for i, y_pred in enumerate(ada_discrete.staged_predict(X_train)):
        ada_discrete_err_train[i] = zero_one_loss(y_pred, y_train)
        
    ax.plot(np.arange(n_estimators) + 1, ada_discrete_err,
        label='AdaBoost Test Error',
        color='red')
    ax.plot(np.arange(n_estimators) + 1, ada_discrete_err_train,
            label='AdaBoost Train Error',
            color='blue')
    
    ax.set_ylim((0.0, 0.5))
    ax.set_xlabel('n_estimators')
    ax.set_ylabel('error rate')

    leg = ax.legend(loc='upper right', fancybox=True)
    leg.get_frame().set_alpha(0.7)
    plt.savefig(f"../../visualization/{sys.argv[1]}.png")