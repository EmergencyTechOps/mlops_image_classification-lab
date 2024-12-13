from flask import Flask, request, render_template, redirect, url_for
import boto3
import tensorflow as tf
from PIL import Image
import numpy as np
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Initialize S3 and download model
s3 = boto3.client('s3')
bucket_name = 'mlops-image-dataset'
model_path = '/tmp/model.h5'

if not os.path.exists(model_path):
    print("Downloading model from S3...")
    s3.download_file(bucket_name, 'model/model.h5', model_path)

# Load the model
model = tf.keras.models.load_model(model_path)

def preprocess_image(image_path):
    """Preprocess the uploaded image for model prediction."""
    img = Image.open(image_path).resize((150, 150))
    img_array = np.array(img) / 255.0  # Normalize the image
    return img_array.reshape(1, 150, 150, 3)

@app.route('/')
def index():
    """Render the upload page."""
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction."""
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400

    # Save uploaded file
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    file.save(file_path)

    # Preprocess and predict
    img_array = preprocess_image(file_path)
    predictions = model.predict(img_array)

    # Create response based on predictions
    label = 'Object Detected' if predictions[0][0] > 0.5 else 'No Object Detected'
    return render_template('result.html', label=label, image_url=file_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
