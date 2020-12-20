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



if __name__ == "__main__":
    loaded_model = joblib.load(f"../../storage/models/{sys.argv[1]}.joblib")
    
    x = "Distance(mi),Side,Temperature(F),Wind_Chill(F),Humidity(%),Pressure(in),Visibility(mi),Wind_Speed(mph),Precipitation(in),Clear,Cloud,Rain,Heavy_Rain,Snow,Heavy_Snow,Fog,Junction,Crossing,Traffic_Signal,Sunrise_Sunset,Duration,Wind_Direction_E,Wind_Direction_N,Wind_Direction_NE,Wind_Direction_NW,Wind_Direction_S,Wind_Direction_SE,Wind_Direction_SW,Wind_Direction_VAR,Wind_Direction_W,TimeofDay_Early Morning,TimeofDay_Evening,TimeofDay_Morning,Season_Spring,Season_Summer,Season_Winter,Day_Type_Weekend".split(',')
    fig = plt.figure(figsize=(25,20))
    _ = tree.plot_tree(loaded_model.best_estimator_, 
                       feature_names=x,  
                       class_names=['Severity_1','Severity_2','Severity_3','Severity_4'],
                       filled=True)
    
    fig.savefig(f"../../visualization/{sys.argv[1]}.png")
    