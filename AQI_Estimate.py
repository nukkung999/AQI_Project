import sklearn
from sklearn.metrics import r2_score 
import joblib
import pandas as pd
import numpy as np

x_test = pd.read_csv(".\\x_test.csv")

AQI_test = (x_test.AQI).to_list()
label_test = (x_test.label).to_list()
x_test = x_test.drop(columns = ['tmp_ind', 'label', 'AQI'])

class_model = joblib.load("path_to_classification_model_file")

reg_vg_model = joblib.load("path_to_very good_model_file")
reg_g_model = joblib.load("path_to_good_model_file")
reg_s_model = joblib.load("path_to_satisfactory_model_file")
reg_uh_model = joblib.load("path_to_unhealthy_model_file")

class_pred = class_model.predict(x_test)
reg_vg_pred = reg_vg_model.predict(x_test)
reg_g_pred = reg_g_model.predict(x_test)
reg_s_pred = reg_s_model.predict(x_test)
reg_uh_pred = reg_uh_model.predict(x_test)

pred_result = {
    "Class" : label_test,
    "Class_Pred" : class_pred,
    "AQI" : AQI_test,
    "VG_Model" : reg_vg_pred,
    "G_Model" : reg_g_pred,
    "S_Model" : reg_s_pred,
    "UH_Model" : reg_uh_pred
}

pred = pd.DataFrame(pred_result)

AQI_pred = []

for index, data in pred.iterrows():
    if data.Class_Pred == 1 :
        AQI_pred.append(data.VG_Model)
    elif data.Class_Pred == 2 :
        AQI_pred.append(data.G_Model)
    elif data.Class_Pred == 3 :
        AQI_pred.append(data.S_Model)
    else :
        AQI_pred.append(data.UH_Model)

print(r2_score(AQI_test, AQI_pred))