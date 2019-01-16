from flask import Flask, render_template, url_for, flash, redirect
from flask_bootstrap import Bootstrap
from flask_datepicker import datepicker
from bson import ObjectId # For ObjectId to work
from flask_pymongo import PyMongo
import os
from flask_fontawesome import FontAwesome

IMAGE_FOLDER = os.path.join('static', 'image')
app = Flask(__name__)

Bootstrap(app)
datepicker(app)
fa = FontAwesome(app)

app.config["MONGO_DBNAME"]= "TG_database"
app.config["MONGO_URI"] = "mongodb://localhost:27017/TG_database"
app.config['UPLOAD_FOLDER'] = IMAGE_FOLDER 
mongo = PyMongo(app)

@app.route("/")
@app.route("/home")
def home():
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'logo.png')
    results = mongo.db.date_collection.find({"guard_exist": True})
    return render_template('home.html', results=results, logo_image = full_filename)

@app.route("/table")
def table():
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'logo.png')
    results = mongo.db.date_collection.find({"guard_exist": True})
    return render_template('table.html', results=results, logo_image = full_filename, title='Data Table')

if __name__ == '__main__':
    app.run(debug=True)