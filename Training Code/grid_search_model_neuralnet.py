from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
import joblib
import numpy as np
import pandas as pd

if __name__ == "__main__" :
    X_train = pd.read_csv("../../storage/data/processed/traning_data.csv")
    X_test = pd.read_csv("../../storage/data/processed/testing_data.csv")
    y_train = np.load("../../storage/data/processed/y_train.npy")
    y_test = np.load("../../storage/data/processed/y_test.npy")
    print("data loaded")
        
    model = GridSearchCV(estimator = MLPClassifier(), param_grid = {'hidden_layer_sizes':((100,),(246, 128, 64)), 'early_stopping':(True,)}, n_jobs = 5).fit(X_train,y_train)
    joblib.dump(model,"../../storage/models/Neural_Nets_Grid.joblib")