from flask import Flask,render_template,request,redirect,jsonify
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application=Flask(__name__)
app=application

ridge_model=pickle.load(open("models/model.pkl","rb"))
scaler_model=pickle.load(open("models/scaler.pkl","rb"))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predictdata",methods=["GET","POST"])
def predict_datapoint():
    if request.method=="POST":
        Temperature=float(request.form.get("Temperature"))
        RH=float(request.form.get("RH"))
        Ws=float(request.form.get("Ws"))
        Rain=float(request.form.get("Rain"))
        FFMC=float(request.form.get("FFMC"))
        DMC=float(request.form.get("DMC"))
        DC=float(request.form.get("DC"))
        ISI=float(request.form.get("ISI"))
        BUI=float(request.form.get("BUI"))
        Classes=float(request.form.get("Classes"))
        Region=float(request.form.get("Region"))

        new_data_scaled=scaler_model.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,DC,ISI,BUI,Classes,Region]])
        result=ridge_model.predict(new_data_scaled)
        return render_template("home.html",result=result[0])
    else:
        return render_template("home.html")

if __name__ == "__main__":
    app.run("0.0.0.0")
