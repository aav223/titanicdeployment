import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd
import joblib
import json

app=Flask(__name__)
#Load the model
RFC_pipeline = joblib.load('best_pipe.pkl')

''' SAMPLE
{
    "data":{
        "pclass":"3",
        "sex":"male",
        "age":"22.00",
        "sibsp":"1",
        "parch":"0",
        "fare":"7.2500",
        "embarked":"S",
        "class":"Third",
        "who":"man",
        "adult_male":"True",
        "deck":"NaN",
        "embark_town":"Southampton",
        "alone":"False"

    }
}'''

@app.route('/')
def home():
    return render_template('home.html') #create a templates named folder and add home.html file

@app.route('/predict_api',methods=['POST']) # For API applications
def predict_api():
    data=request.json['data'] #input data stored here
    df_data = pd.json_normalize(data) # creating dataframe from input
    output=RFC_pipeline.predict(df_data)
    json_numbers = json.dumps(output[0],default=str)
    return json_numbers

    
if __name__=="__main__":
    app.run(debug=True)