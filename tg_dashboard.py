from flask import Flask, render_template, url_for, flash, redirect
from bson import ObjectId # For ObjectId to work
from bson.json_util import dumps
from flask_pymongo import PyMongo

app = Flask(__name__)

app.config["MONGO_DBNAME"]= "TG_database"
app.config["MONGO_URI"] = "mongodb://localhost:27017/TG_database"
mongo = PyMongo(app)

@app.route("/home")
def home():
    results = mongo.db.date_collection.find({"guard_exist": True})
    return render_template('home.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)