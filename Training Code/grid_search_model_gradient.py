from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
import joblib
import numpy as np
import pandas as pd

if __name__ == "__main__" :
    X_train = pd.read_csv("../../storage/data/processed/traning_data.csv")
    X_test = pd.read_csv("../../storage/data/processed/testing_data.csv")
    y_train = np.load("../../storage/data/processed/y_train.npy")
    y_test = np.load("../../storage/data/processed/y_test.npy")
    print("data loaded")
        
    model = GradientBoostingClassifier(verbose = 1).fit(X_train,y_train)
    joblib.dump(model,"../../storage/models/gradient2_grid_search.joblib")
