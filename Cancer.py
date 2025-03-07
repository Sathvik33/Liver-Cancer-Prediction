# import pandas as pd
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,root_mean_squared_error
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import GridSearchCV
# from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
# from sklearn.model_selection import RandomizedSearchCV
# from scipy.stats import uniform
# from sklearn.model_selection import cross_val_score
# from sklearn.metrics import precision_recall_fscore_support
# from imblearn.over_sampling import SMOTE   
# from imblearn.under_sampling import RandomUnderSampler
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.ensemble import BaggingClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.ensemble import GradientBoostingClassifier
# from xgboost import XGBClassifier,XGBRegressor

# data=pd.read_csv(r"C:\C_py\Project\Cancer\data\liver_cancer_prediction.csv")


# label_encoder=LabelEncoder()
# categorical_columns = ['Country', 'Region', 'Population', 'Incidence_Rate', 'Mortality_Rate',
#        'Gender', 'Age', 'Alcohol_Consumption', 'Smoking_Status',
#        'Hepatitis_B_Status', 'Hepatitis_C_Status', 'Obesity', 'Diabetes',
#        'Rural_or_Urban', 'Seafood_Consumption', 'Herbal_Medicine_Use',
#        'Healthcare_Access', 'Screening_Availability', 'Treatment_Availability',
#        'Liver_Transplant_Access', 'Ethnicity', 'Preventive_Care',
#        'Survival_Rate', 'Cost_of_Treatment']

# for i in data.columns:
#     if data[i].dtype == 'object':
#         data[i]=label_encoder.fit_transform(data[i])

# x=data.drop(columns="Prediction")
# y = data["Prediction"].map({"No": 0, "Yes": 1})

# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

# scaler=StandardScaler()
# x_train_sca=scaler.fit_transform(x_train)
# x_test_sca=scaler.transform(x_test)

# smote=SMOTE()
# x_train_scm,y_train_scm=smote.fit_resample(x_train_sca,y_train)

# param_dis={
#     # For XGBClassifier
#     'n_estimators': [100, 200],
#     'learning_rate': [0.01, 0.1, 1],
#     'max_depth': [1,3,5],
#     'subsample': [0.8, 1.0],
#     'colsample_bytree': [0.6,0.8, 1.0],
# }

# model=GridSearchCV(estimator=XGBClassifier(),param_grid=param_dis,cv=5,n_jobs=-1,verbose=1)
# model.fit(x_train_scm,y_train_scm)
# y_pred=model.predict(x_test_sca)
# print(pd.DataFrame({"Actual: ":y_test,"predicted":y_pred}))
# print(f"Accuracy score: {accuracy_score(y_test,y_pred)*100:.2f}%")
# print(f"Recall: {recall_score(y_test,y_pred,average='micro',pos_label=1)*100:.2f}%")
# print(f"f1-score: {f1_score(y_test,y_pred,average='micro',pos_label=1)*100:.2f}%")
# print(f"precision: {precision_score(y_test,y_pred,average='micro',pos_label=1)*100:.2f}%")

# import joblib
# joblib.dump(model.best_estimator_,r'C:\C_py\Project\Cancer\Cancerpred.joblib')
# joblib.dump(label_encoder,r'C:\C_py\Project\Cancer\encoders.joblib')
# joblib.dump(scaler,r'C:\C_py\Project\Cancer\scaler.joblib')



import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import joblib

data = pd.read_csv(r"C:\C_py\Project\Cancer\data\liver_cancer_prediction.csv")

categorical_columns = ['Country', 'Region', 'Gender', 'Alcohol_Consumption', 'Smoking_Status',
                       'Hepatitis_B_Status', 'Hepatitis_C_Status', 'Obesity', 'Diabetes',
                       'Rural_or_Urban', 'Seafood_Consumption', 'Herbal_Medicine_Use',
                       'Healthcare_Access', 'Screening_Availability', 'Treatment_Availability',
                       'Liver_Transplant_Access', 'Ethnicity', 'Preventive_Care']

label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

y = data["Prediction"].map({"No": 0, "Yes": 1}).astype(int)
x = data.drop(columns=["Prediction"])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
x_train_sca = scaler.fit_transform(x_train)
x_test_sca = scaler.transform(x_test)

smote = SMOTE()
x_train_scm, y_train_scm = smote.fit_resample(x_train_sca, y_train)

param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1, 1],
    'max_depth': [1, 3, 5],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
}

model = GridSearchCV(estimator=XGBClassifier(), param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)
model.fit(x_train_scm, y_train_scm)

y_pred = model.predict(x_test_sca)

print(pd.DataFrame({"Actual": y_test, "Predicted": y_pred}))
print(f"Accuracy score: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(f"Recall: {recall_score(y_test, y_pred, average='micro') * 100:.2f}%")
print(f"F1-score: {f1_score(y_test, y_pred, average='micro') * 100:.2f}%")
print(f"Precision: {precision_score(y_test, y_pred, average='micro') * 100:.2f}%")

joblib.dump(model.best_estimator_, r'C:\C_py\Project\Cancer\Cancerpred.joblib')
joblib.dump(label_encoders, r'C:\C_py\Project\Cancer\encoders.joblib')
joblib.dump(scaler, r'C:\C_py\Project\Cancer\scaler.joblib')
