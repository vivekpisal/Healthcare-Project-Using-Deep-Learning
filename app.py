from __future__ import division, print_function
import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf

from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

app=Flask(__name__)



@app.route("/")
def index():
	return render_template("index.html")


@app.route("/corona")
def corona():
	return render_template("corona.html")


@app.route("/tumor")
def tumor():
	return render_template("tumor.html")




def corona_predict(img_path):
    MODEL_PATH ='vgg16.h5'
    model = load_model(MODEL_PATH)
    img = image.load_img(img_path, target_size=(224, 224))

    #Preprocessing the image
    x = image.img_to_array(img)
    #x = np.true_divide(x, 255)
    x=x/255
    x = np.expand_dims(x, axis=0)
   

    preds1= model.predict(x)
    preds=np.argmax(preds1, axis=1)
    if preds==0:
        preds="Negative Patient"
    elif preds==1:
        preds="Positive Patient"
    return preds,preds1



def tumor_predict(img_path):
    MODEL_PATH ='Tumorvgg16.h5'
    model = load_model(MODEL_PATH)
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    ## Scaling
    x=x/255
    x = np.expand_dims(x, axis=0)
    preds1= model.predict(x)
    preds=np.argmax(preds1, axis=1)
    if preds==0:
        preds="Glioma Tumor"
    elif preds==1:
        preds="Meningioma Tumor"
    elif preds==2:
    	preds="No Tumor"
    elif preds==3:
    	preds="Pituitary Tumor"
    return preds,preds1




@app.route('/predict', methods=['GET', 'POST'])
def upload_corona():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds,preds1= corona_predict(file_path)
        result=preds
        print(preds1)
        return render_template("corona.html",result=result,x=1)
    return redirect("/corona")




@app.route('/predicttumor', methods=['GET', 'POST'])
def upload_tumor():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads',f.filename)
        f.save(file_path)

        # Make prediction
        result,preds1= tumor_predict(file_path)
        return render_template("tumor.html",result=result,x=1,file_path=file_path)
    return redirect("/tumor")




if __name__ == '__main__':
	app.run(debug=True)