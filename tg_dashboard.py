from flask import Flask, render_template, url_for, flash, redirect, request
from flask_bootstrap import Bootstrap
from bson import ObjectId # For ObjectId to work
from flask_pymongo import PyMongo
from flask_fontawesome import FontAwesome
from flask_wtf import Form
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
        start_date = request.form['dateSelect']
        print(start_date)
        return render_template('home.html', results=results, logo_image=full_filename, start_date=start_date, start=start)
        
    else:
        return render_template('home.html', results=results, logo_image=full_filename)

@app.route("/table")
def table():
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'logo.png')
    results = mongo.db.date_collection.find({"guard_exist": True})
    return render_template('table.html', results=results, logo_image = full_filename, title='Data Table')

if __name__ == '__main__':
    app.run(debug=True)