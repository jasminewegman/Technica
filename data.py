#imports
#Credits:
#link to the data used to train AI model: https://www.kaggle.com/datasets/willianoliveiragibin/healthcare-insurance

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


kaggle_data=pd.read_csv("C:\\Users\\jasmi\\OneDrive\\Documents\\Technica\\final project\\insurance.csv")
print(kaggle_data.head())

#preprocessing
demographic_columns=['age','sex', 'bmi', 'children', 'smoker', 'region']
output_target_column='charges'
#quantitative specific
quan=['age','bmi','children']
num_transform=StandardScaler()
#qualitative specific (OHE preprocessing)
qual=['sex','smoker','region']
qual_transform=OneHotEncoder() 
#set axis
x=kaggle_data[demographic_columns]
y=kaggle_data[output_target_column]
#splitting
x_tr,x_te,y_tr,y_te=train_test_split(x,y,test_size=.2,random_state=42)

#transforming colums for preprocessing
preprocessor=ColumnTransformer(
    transformers=[
        ('quantitative', StandardScaler(), quan), #quan data
        ('qualitative', OneHotEncoder(handle_unknown='ignore'), qual) #qual data
    ])
#applying model (linear regression) & train to fit
model_pipe=Pipeline(steps=[
    ('preprocessor',preprocessor),
    ('model',LinearRegression())

])
model_pipe.fit(x_tr,y_tr)
#insurance cost prediction (test set)
y_p=model_pipe.predict(x_te)
#eval metrics (abs and sq err)
abs_err=mean_absolute_error(y_te, y_p)
sq_err=mean_squared_error(y_te,y_p)
cod=r2_score(y_te,y_p)
print('Mean Absolute Err: ', abs_err,'\nMean Squared Err: ', sq_err,'\nCoefficient of Determination: ', cod)

#convert & save using joblib
joblib.dump(model_pipe, 'technica_insurance_model_wegman.pkl')

