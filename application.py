from flask import Flask, render_template, request, send_from_directory
from PIL import Image
import os
#from werkzeug.exceptions import RequestEntityTooLarge

from utils import pre_process
from model_1 import get_model, pred
#import logging

application = Flask(__name__)

app = application

UPLOAD_FOLDER = 'images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # set max upload size to 50 MB

# Configure logging
#logging.basicConfig(filename='app.log', level=logging.DEBUG)


@app.route("/", methods= ['GET','POST'])

def upload():

    if request.method=='POST':

        try:

            image = request.files['image']
            file_name = image.filename

            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)


            #file_path = os.join("/config/workspace/images",image)

            #im = Image.open(image)
            image.save(file_path)

            Image.open(file_path)

        except:
            if 'file_path' in locals():
                if len(file_path.split(".")) > 1:
                    os.remove(file_path)
            error = "Input should be an image"

            return render_template("index.html",error = error)



        return render_template("index.html",addrss = file_name)

    else :
        return render_template("index.html")


@app.route('/uploads/<path:path>')
def serve_image(path):
    error = "Input should be an image within 16 MB"
    return send_from_directory(app.config['UPLOAD_FOLDER'], path)


@app.route('/predict/<filename>', methods=['POST','GET'])
def predict(filename):

    #file_name = request.args.get('path') #When method is get
    #file_name = request.form['path']
    
    image_address = os.path.join(app.config['UPLOAD_FOLDER'], filename)
   
    image_tensors = pre_process(image_address)
    

    tflite_model, input_details, output_details = get_model()
   
    
    result = pred(image_tensors, tflite_model, input_details, output_details)

    return render_template("index.html", result = result,addrss = filename )

    #return str(result)


if __name__ == '__main__':

    app.run(host='127.0.0.1', port=8000, debug=True)
