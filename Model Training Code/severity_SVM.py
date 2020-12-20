import pandas as pd

from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from joblib import dump
        
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', None)

if __name__ == "__main__":
    
    df_FF_ML = pd.read_csv("../../storage/data/intermediate/cleaned_datafolium_3.csv")
    print("Data Loaded")
    target='Severity'

    y=df_FF_ML[target]
    X=df_FF_ML.drop(target, axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto')).fit(X_train, y_train)
    
    dump(clf, '../../storage/models/SVM_1.joblib') 
    
