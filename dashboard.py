from flask import Flask, render_template, url_for, flash, redirect, request
from flask_bootstrap import Bootstrap
from bson import ObjectId # For ObjectId to work
from flask_pymongo import PyMongo
from flask_fontawesome import FontAwesome
from flask_wtf import Form
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import os

IMAGE_FOLDER = os.path.join('static', 'image')
app = Flask(__name__)

Bootstrap(app)
fa = FontAwesome(app)

app.config["MONGO_DBNAME"]= "TG_database"
app.config["MONGO_URI"] = "mongodb://localhost:27017/TG_database"
app.config['UPLOAD_FOLDER'] = IMAGE_FOLDER 
mongo = PyMongo(app)

@app.route("/", methods=['GET','POST'])
@app.route("/home", methods=['GET','POST'])
def home():
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'logo.png')   
    results = mongo.db.date_collection.find({"guard_exist": True})

    if request.method=='POST':
        date = request.form['dateSelect']
        date1 = datetime.strptime(date, "%Y-%m-%d")
        date2 = date1+timedelta(days=1)
        results = mongo.db.date_collection.find({
            "time": {
                "$gte": date1,
                "$lte": date2
            }
        })

        totalDates = []
        for result in results:
            totalDates.append(result["time"])
        
        rng = pd.date_range(date1, periods=24, freq='H')
        rngHours = pd.DatetimeIndex(rng).hour

        # group dates by hour
        # first, convert all the dates into DatetimeIndex, then get all hours of all dates
        uniqueHours = pd.DatetimeIndex(totalDates).hour

        # while rngHour != uniqueHours:

        # second, find the number of occurance of each hour
        # both hours and counts is an array
        hours, counts = np.unique(uniqueHours, return_counts=True)
        zeroes = np.zeros(hours[0]).astype(np.uint8) # create an array of zeroes
        results = np.append(zeroes, counts) # append the count after zeroes

        # fill up the rest with 0
        rest = 24 - len(results)
        results = np.append(results, np.zeros(rest).astype(np.uint8))
        print(results)

        return render_template('home.html', labels=rngHours, values=results, logo_image=full_filename, title='Graph')
        
    else:
        return render_template('home.html', results=results, logo_image=full_filename)

@app.route("/table")
def table():
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'logo.png')
    results = mongo.db.date_collection.find({"guard_exist": True})
    return render_template('table.html', results=results, logo_image = full_filename, title='Data Table')

if __name__ == '__main__':
    app.run(debug=True)