from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
import joblib
import numpy as np
import pandas as pd

if __name__ == "__main__" :
    X_train = pd.read_csv("../../storage/data/processed/traning_data.csv")
    X_test = pd.read_csv("../../storage/data/processed/testing_data.csv")
    y_train = np.load("../../storage/data/processed/y_train.npy")
    y_test = np.load("../../storage/data/processed/y_test.npy")
    print("data loaded")
        
    model = GridSearchCV(estimator = AdaBoostClassifier(), param_grid = {'n_estimators':(40, 50, 80,100,125,150,200,250)}, n_jobs = 5).fit(X_train,y_train)
    joblib.dump(model,"../../storage/models/ADA_Boost_GridSearch.joblib")

