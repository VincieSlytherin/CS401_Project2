from flask import Flask
from flask import request
from flask import jsonify
import pickle
import numpy as np
import os.path, time
import datetime

app = Flask(__name__)
with open('model_v1.pickle', 'rb') as f:    
    app.version=pickle.load(f)
    app.time = pickle.load(f)#time
    app.model=pickle.load(f)#model

app.vectorizer=pickle.load(open("vectorizer.pickle","rb"))
app.tf_transformer=pickle.load(open("tf_transformer.pickle","rb"))

@app.route("/api/american",methods=["POST"])
def predict_country():   
    content = request.get_json(force=True)      
    text=np.array([content["text"]])
    text1=app.vectorizer.transform(text)
    text2=app.tf_transformer.transform(text1)
    predicted = app.model.predict(text2)
    prediction=predicted.tolist()

    return jsonify({"is_american":str(predicted[0]),"version":app.version,"model_date":app.time})
