from flask import Flask, render_template, request
import joblib
from joblib import *
import sklearn
import pickle
import pandas as pd
import numpy as np
app = Flask(__name__)

model = pickle.load(open("flights.pkl", "rb"))

@app.route('/')
def auto():
	return render_template("main.html")

@app.route('/predict', methods = ["POST"])
def home():
	X1 =  request.form['X1']
	X2 =  request.form['X2']
	X3 =  request.form['X3']
	X4 =  request.form['X4']
	X5 =  request.form['X5']
	pred = model.predict(np.array([X1, X2, X3, X4, X5]).reshape(1, -1))
	if (pred <= 0):
		return render_template('main.html',prediction_text="Delay = {} minutes".format("0"))
	else:
		return render_template('main.html',prediction_text="Delay = {} minutes".format(pred))


if (__name__ == "__main__"):
	app.run(debug=True)