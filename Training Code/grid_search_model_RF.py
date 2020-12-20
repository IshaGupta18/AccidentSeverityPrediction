from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import joblib
import numpy as np
import pandas as pd

if __name__ == "__main__" :
    X_train = pd.read_csv("../../storage/data/processed/traning_data.csv")
    X_test = pd.read_csv("../../storage/data/processed/testing_data.csv")
    y_train = np.load("../../storage/data/processed/y_train.npy")
    y_test = np.load("../../storage/data/processed/y_test.npy")
    print("data loaded")
        
    model = GridSearchCV(estimator = RandomForestClassifier(verbose = 3), param_grid = {'n_estimators': (10,30,50,70,90,100,125,135,150,160)}, n_jobs =1).fit(X_train,y_train)
    joblib.dump(model,"../../storage/models/Random_forest2_grid_search.joblib")
