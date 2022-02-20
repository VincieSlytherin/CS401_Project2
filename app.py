#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask
from flask import request
from flask import jsonify
import pickle
import numpy as np
import os.path, time
import datetime



# In[ ]:

app = Flask(__name__)

app.model=pickle.load(open("model_v1.pickle","rb"))
app.vectorizer=pickle.load(open("vectorizer.pickle","rb"))
app.tf_transformer=pickle.load(open("tf_transformer.pickle","rb"))
app.filePath = "model_v1.pickle"
app.ModifiedTime=time.localtime(os.stat(app.filePath).st_mtime) #文件访问时间 
app.time=time.strftime("%Y-%m-%d-%H:%M:%S",app.ModifiedTime)


@app.route("/api/american",methods=["POST"])


def predict_country():
    
    content = request.get_json(force=True) 
    if type(content["text"])!=str:
        raise TypeError
#         return None
    else:
       
        text=np.array([content["text"]])

        text1=app.vectorizer.transform(text)
        text2=app.tf_transformer.transform(text1)
        predicted = app.model.predict(text2)
        prediction=predicted.tolist()
        result=""
#         for i in predicted:
#             result+=str(i)+" "

    #     filePath = "/home/rj133/Project2/model_v1.pickle"
    #     ModifiedTime=time.localtime(os.stat(filePath).st_mtime) #文件访问时间 
    #     d2=time.strftime("%Y-%m-%d-%H:%M:%S",ModifiedTime)

    return jsonify({"is_american":str(predicted[0]),"version":"MultinomialNB_0","model_date":app.time})
