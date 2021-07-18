from flask import Flask,request,render_template,jsonify
from flask_cors import CORS,cross_origin
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

@app.route('/',methods=['GET'])  # route to display the home page
@cross_origin()
def index():
    return render_template("index.html")

@app.route('/predict',methods=['GET','POST'])
def home():

    if request.method== 'POST':
        try:
            occ_2 = float(request.form['occ_2'])
            occ_3 = float(request.form['occ_3'])
            occ_4 = float(request.form['occ_4'])
            occ_5 = float(request.form['occ_5'])
            occ_6 = float(request.form['occ_6'])
            occ_husb_2 = float(request.form['occ_husb_2'])
            occ_husb_3 = float(request.form['occ_husb_3'])
            occ_husb_4 = float(request.form['occ_husb_4'])
            occ_husb_5 = float(request.form['occ_husb_5'])
            occ_husb_6 = float(request.form['occ_husb_6'])
            rate_marriage = float(request.form['rate_marriage'])
            age= float(request.form['age'])
            yrs_married= float(request.form['yrs_married'])
            children= float(request.form['children'])
            religious= float(request.form['religious'])
            educ= float(request.form['educ'])
            model=pickle.load(open('logisticModel.sav','rb'))
            scaler = pickle.load(open('scaler.sav','rb'))

            pred = model.predict(scaler.transform([[occ_2,occ_3,occ_4,occ_5,occ_6,occ_husb_2,occ_husb_3,occ_husb_4,occ_husb_5,occ_husb_6,rate_marriage,age,yrs_married,children,religious,educ]]))

            result = ''
            color = ''
            if pred == 1:
                result='YES'
                color='red'
            else:
                color='green'
                result= 'NO'
            return  render_template("result.html",result=result,color= color)

        except Exception as e:
            print(e)
            return "Something Wrong"
    else:
        return render_template("index.html")


if __name__=="__main__":
    app.run(debug=True)
