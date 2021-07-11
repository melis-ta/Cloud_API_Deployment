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
model=pickle.load(open('model.pkl','rb'))


# In[3]:


@app.route('/')
def home():
        return render_template('index.html')


# In[4]:


@app.route('/predict', methods=['POST'])
def predict():
    
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = Diabetes_model.predict(final_features)
    
    output = round(prediction[0], 2)
    
    return (flask.render_template('index.html', prediction_text='The probability that the patient has diabetes is {}'.format(output)))


# In[ ]:


if __name__=='__main__':
    app.run(debug=True, use_reloader=False)


# In[ ]:




