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


def test_classifier(clf):
    print(clf)
    X_train = pd.read_csv("../../storage/data/processed/traning_data.csv")
    X_test = pd.read_csv("../../storage/data/processed/testing_data.csv")
    y_train = np.load("../../storage/data/processed/y_train.npy")
    y_test = np.load("../../storage/data/processed/y_test.npy")

    

    print("Test Data :\n")
    predictions=loaded_model.predict(X_test)
    print("[Logistic Regression algorithm] accuracy_score:",accuracy_score(y_test, predictions))
    print('\n')
    print('Confusion Matrix of results')
    print(confusion_matrix(y_test,predictions))
    print('\n')
    print('Classificaiton Report of results')
    print(classification_report(y_test,predictions))
    
    print("Trainin Data :\n")

    predictions=loaded_model.predict(X_train)
    print("[Logistic Regression algorithm] accuracy_score:",accuracy_score(y_train, predictions))
    print('\n')
    print('Confusion Matrix of results')
    print(confusion_matrix(y_train,predictions))
    print('\n')
    print('Classificaiton Report of results')
    print(classification_report(y_train,predictions))
    
    
if __name__ == "__main__":
    loaded_model = joblib.load(f"../../storage/models/{sys.argv[1]}.joblib")
    test_classifier(loaded_model)
