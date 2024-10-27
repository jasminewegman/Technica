#i named this the wrong thing, whoops (it was too late to turn back (its 7:49am rn))! this file runs the application in flask 


import pandas as pd
import joblib 
from flask import Flask, request, jsonify, render_template
#data.py import
model_pipe=joblib.load("C:\\Users\\jasmi\\OneDrive\\Documents\\Technica\\final\\technica_insurance_model_wegman.pkl")

app=Flask('Insurance_Model')
@app.route('/')
def about():
    return render_template('about.html')
@app.route('/index', methods=['GET'])
def index():
    return render_template('index.html')
@app.route('/predict',methods=['POST'])
def pred():
    try:
        age=request.form.get('age',type=int)
        sex=request.form.get('sex')
        bmi=request.form.get('bmi', type=float)
        children=request.form.get('children', type=int)
        smoker=request.form.get('smoker')
        region=request.form.get('region')

        input_data=pd.DataFrame({
            'age':[age],
            'sex':[sex],
            'bmi':[bmi],
            'children':[children],
            'smoker':[smoker],
            'region':[region]
        })
        input_val(input_data)
        pred_output=model_pipe.predict(input_data)
        return jsonify(prediction=pred_output.tolist())
    except Exception as e:
        return jsonify(error=str(e)),400
#check for err
def input_val(df):
    ex_col=['age','sex','bmi','children','smoker','region']
    if not all(column in df.columns for column in ex_col):
        raise ValueError('Required input data is missing.')
    real_cat={
        'sex':['deffemale','male'],
        'smoker':['yes','no'],
        'region':['northeast','northwest','southeast','southwest']
    }
    for column in ['sex','smoker','region']:
        if any(item not in real_cat[column] for item in df[column]):
            raise ValueError(f'Invalid categort in column {column}')

if __name__=='__main__':
    app.run(debug=True)