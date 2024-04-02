from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import keras

app = Flask(__name__)

# Set the path where uploaded images will be stored
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the pre-trained model for potato
potato_model = tf.keras.models.load_model("my_model.h5")
potato_class_names = ["Potato___Early_blight", "Potato___Late_blight", "Potato___healthy"]

# Load the pre-trained model for tomato
tomato_model = keras.models.load_model('cnn.h5')
tomato_class_labels = ["Tomato_Bacterial_spot", "Early_blight", "Late_blight", "Leaf_Mold", "Septoria_leaf_spot",
                       "Spider_mites Two-spotted_spider_mite", "Target_Spot", "Tomato_Yellow_Leaf_Curl_Virus",
                       "Tomato_mosaic_virus", "Tomato___healthy"]


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_potato(uploaded_image):
    img = Image.open(uploaded_image)
    img = img.resize((224, 224))
    img = img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    predictions = potato_model.predict(img)
    predicted_class = potato_class_names[np.argmax(predictions)]
    confidence = round(100 * np.max(predictions), 2)

    return predicted_class, confidence


def predict_tomato(uploaded_image):
    img = Image.open(uploaded_image)
    img = img.resize((224, 224))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = tomato_model.predict(img_array)
    predicted_class_index = np.argmax(prediction)
    predicted_class = tomato_class_labels[predicted_class_index]

    return predicted_class


@app.route('/')
def homepage():
    return render_template('homepage.html')


@app.route('/potato_app', methods=['GET', 'POST'])
def run_potato_app():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            predicted_class, confidence = predict_potato(file_path)

            return render_template('result.html', vegetable='Potato', image_path=file_path,
                                   predicted_class=predicted_class, confidence=confidence)

    return render_template('potato_app.html')


@app.route('/tomato_app', methods=['GET', 'POST'])
def run_tomato_app():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            predicted_class = predict_tomato(file_path)

            return render_template('result.html', vegetable='Tomato', image_path=file_path,
                                   predicted_class=predicted_class)

    return render_template('tomato_app.html')


if __name__ == "__main__":
    app.run(debug=True)
