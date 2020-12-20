from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import joblib
import numpy as np
import pandas as pd

def data_load():
	X_train = pd.read_csv("../../storage/data/processed/traning_data.csv")
	# X_test = pd.read_csv("../../storage/data/processed/testing_data.csv")
	y_train = np.load("../../storage/data/processed/y_train.npy")
	# y_test = np.load("../../storage/data/processed/y_test.npy")

	return X_train, y_train


if __name__ == "__main__" :


	c = [
			["Distance(mi)"],
			["Side"],
			["Temperature(F)","Humidity(%)","Pressure(in)","Visibility(mi)","Precipitation(in)"],
			["Wind_Chill(F)","Wind_Speed(mph)","Wind_Direction_E","Wind_Direction_N","Wind_Direction_NW","Wind_Direction_S","Wind_Direction_SE","Wind_Direction_SW","Wind_Direction_VAR","Wind_Direction_W"],
			["Clear","Cloud","Rain","Heavy_Rain","Snow","Heavy_Snow","Fog",],
			["Junction","Crossing","Traffic_Signal",],
			["Sunrise_Sunset",],
			["Duration",],
			["TimeofDay_Early Morning","TimeofDay_Evening","TimeofDay_Morning",],
			["Season_Spring","Season_Summer","Season_Winter",],
			["Day_Type_Weekend"],
	]
	for i in range(len(c)):
		print(f"iteration {i}")
		X_train, y_train = data_load()

		X_train.drop(columns=c[i])

		print(f"data loaded {i}")
		model = RandomForestClassifier(verbose = 3, n_estimators = 150, n_jobs = 20).fit(X_train,y_train)
		joblib.dump(model,f"../../storage/models/rf_features_{i}.joblib")
		

