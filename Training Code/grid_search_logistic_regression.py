from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
import numpy as np
import pandas as pd

if __name__ == "__main__" :
    X_train = pd.read_csv("../../storage/data/processed/traning_data.csv")
    X_test = pd.read_csv("../../storage/data/processed/testing_data.csv")
    y_train = np.load("../../storage/data/processed/y_train.npy")
    y_test = np.load("../../storage/data/processed/y_test.npy")
    print("data loaded")
    #     pipe = Pipeline([('classifier' , LogisticRegression(verbose = 1))])

    #     grid = (
    #         {
    #             'classifier' : [LogisticRegression(verbose = 1)],
    #             'classifier__penalty' : ['l1', 'l2'],
    #             'classifier__solver' : ['saga'],
    #             'classifier__C' : np.logspace(-4, 4, 20), // 10**-4 to 10**4
    #             'classifier__max_iter' : [1000]
    #         },
    #         {
    #             'classifier' : [LogisticRegression(verbose = 1)],
    #             'classifier__penalty' : ['l1'],
    #             'classifier__solver' : ['sag'],
    #             'classifier__C' : np.logspace(-4, 4, 20),
    #             'classifier__max_iter' : [1000]
    #         }
    #     )
    #model = GridSearchCV(pipe, param_grid = grid, n_jobs = 25).fit(X_train,y_train)
    model = LogisticRegression(C=545.5594781168514, max_iter=1000, penalty='l1',
                   solver='saga', verbose=10).fit(X_train,y_train)
    joblib.dump(model,"../../storage/models/logistic2_grid_search.joblib")

