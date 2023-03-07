import numpy as np
import pandas as pd
import joblib
from joblib import dump, load
import sklearn
from sklearn import linear_model, tree
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier, DistanceMetric
from sklearn.metrics import confusion_matrix, mean_absolute_error, classification_report
from sklearn.model_selection import train_test_split

data_file = ".\\DataNew.csv"

data = pd.read_csv(data_file)
data.dropna(inplace=True)

label = []

for index, elem in data.iterrows():
    AQI = elem.AQI.astype(int)

    if AQI <= 25 :
        label.append(1)
    elif AQI <= 50 :
        label.append(2)
    elif AQI <= 100 :
        label.append(3)
    else :
        label.append(4)

data["Label"] = label

vg_data = data.copy()
g_data = data.copy()
s_data = data.copy()
uh_data = data.copy()

vg_data.drop(index = vg_data[vg_data["AQI"] > 25].index, inplace = True)
# Drop data that AQI > 25 --> Very Good range is 0 - 25

g_data.drop(index = g_data[g_data["AQI"] < 26].index, inplace = True)
g_data.drop(index = g_data[g_data["AQI"] > 50].index, inplace = True)
# Drop data that AQI < 26 and AQI > 50 Good range is 26 - 50

s_data.drop(index = s_data[s_data["AQI"] < 51].index, inplace = True)
s_data.drop(index = s_data[s_data["AQI"] > 100].index, inplace = True)
# Drop data that AQI < 51 and AQI > 100 Satisfactory range is 51 - 100

uh_data.drop(index = uh_data[uh_data["AQI"] < 101].index, inplace = True)
# Drop data that AQI < 101 Unhealthy range is 101+

y_vg = vg_data[["AQI", "Label"]]
x_vg = vg_data.drop(columns = ["AQI", "Label"])

y_g = g_data[["AQI", "Label"]]
x_g = g_data.drop(columns = ["AQI", "Label"])

y_s = s_data[["AQI", "Label"]]
x_s = s_data.drop(columns = ["AQI", "Label"])

y_uh = uh_data[["AQI", "Label"]]
x_uh = uh_data.drop(columns = ["AQI", "Label"])

# We use 70/30 test_size we fix to 0.3
x_vg_train, x_vg_test, y_vg_train, y_vg_test = train_test_split(x_vg, y_vg, test_size = 0.3) 
x_g_train, x_g_test, y_g_train, y_g_test = train_test_split(x_g, y_g, test_size = 0.3)
x_s_train, x_s_test, y_s_train, y_s_test = train_test_split(x_s, y_s, test_size = 0.3)
x_uh_train, x_uh_test, y_uh_train, y_uh_test = train_test_split(x_uh, y_uh, test_size = 0.3)

# Merge each class data to x_test
# Data of each class leave it. We will use it on regressor
x_train = x_vg_train.copy()
x_train = x_train.append(x_g_train, ignore_index = True)
x_train = x_train.append(x_s_train, ignore_index = True)
x_train = x_train.append(x_uh_train, ignore_index = True)

x_test = x_vg_test.copy()
x_test = x_test.append(x_g_test, ignore_index = True)
x_test = x_test.append(x_s_test, ignore_index = True)
x_test = x_test.append(x_uh_test, ignore_index = True)

y_train = y_vg_train.copy()
y_train = y_train.append(y_g_train, ignore_index = True)
y_train = y_train.append(y_s_train, ignore_index = True)
y_train = y_train.append(y_uh_train, ignore_index = True)

y_test = y_vg_test.copy()
y_test = y_test.append(y_g_test, ignore_index = True)
y_test = y_test.append(y_s_test, ignore_index = True)
y_test = y_test.append(y_uh_test, ignore_index = True)

# Concat data for save to file
train_file = pd.concat([x_train, y_train], axis = 1)
test_file = pd.concat([x_test, y_test], axis = 1)
train_file.to_csv('x_train.csv')
test_file.to_csv('x_test.csv')

class_model = KNeighborsClassifier(n_neighbors = 1)
class_model.fit(x_train, y_train)
joblib.dump(class_model, "Classification.model")

reg_vg_model = KNeighborsRegressor(n_neighbors = 1)
reg_vg_model.fit(x_vg_train, y_vg_train)
joblib.dump(reg_vg_model, "VeryGood_Regressor.model")

reg_g_model = KNeighborsRegressor(n_neighbors = 1)
reg_g_model.fit(x_g_train, y_g_train)
joblib.dump(reg_g_model, "Good_Regressor.model")

reg_s_model = KNeighborsRegressor(n_neighbors = 1)
reg_s_model.fit(x_s_train, y_s_train)
joblib.dump(reg_s_model, "Satisfactory_Regressor.model")

reg_uh_model = tree.DecisionTreeRegressor(criterion = "mae")
reg_uh_model.fit(x_uh_train, y_uh_train)
joblib.dump(reg_s_model, "Unhealthy_Regressor.model")