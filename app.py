#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing necessary libraries
import pandas as pd
import flask
from flask import Flask, jsonify, request, render_template
import pickle


# In[2]:


#creating flask app
app= Flask(__name__)


# In[3]:


@app.route('/', methods=['GET', 'POST'])
def home():
    if (request.method== 'GET'):
        data= "Enter your test results!"
        return jsonify ({'data': data})


# In[4]:


@app.route('/predict/', endpoint='Diabetes_prediction')
def Diabetes_prediction():
    model=pickle.load(open('model.pickle','rb'))
    Pregnancies= request.args.get('Pregnancies')
    Glucose=request.args.get('Glucose')
    BloodPressure=request.args.get('BloodPressure')
    SkinThickness=request.args.get('SkinThickness')
    Insulin=request.args.get('Insulin')
    BMI=request.args.get('BMI')
    DiabetesPedigreeFunction=request.args.get('DiabetesPedigreeFunction')
    Age=request.args.get('Age')
    
    test_df= pd.DataFrame({' Pregnancies': [Pregnancies], 'Glucose': [Glucose], 
                           'BloodPressure': [BloodPressure], 'SkinThickness': [SkinThickness], 
                           'Insulin': [Insulin], 'BMI': [BMI], 
                           'DiabetesPedigreeFunction': [DiabetesPedigreeFunction], 'Age': [Age]})
    pred_diabetes= model.predict(test_df)
    return jsonify({'Diabetes Predicted': str(pred_diabetes)})


# In[ ]:


if __name__=='__main__':
    app.run(debug=True, use_reloader=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




