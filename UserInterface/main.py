from flask import Flask, render_template, request, jsonify, redirect
from keras.models import load_model
import numpy as np
import cv2
from PIL import Image
import os
import subprocess

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__)

# Load models once at the beginning
models = {
    'lenet5': load_model('../Models/LeNet5/50lenet5.keras'),
    'ran2dev': load_model('../Models/Ran2Dev/50ran2dev.keras')
}

# Class labels for Ranjana script
class_labels = [
    'अ', 'आ', 'अ:', 'ऐ', 'अं', 'औ', 'ब', 'भ', 'च', 'छ', 'ड',
    'द', 'ध', 'ढ', 'ए', '८', '५', '५', 'ग', 'घ', 'ज्ञ',
    'ह', 'इ', 'ई', 'ज', 'झ', 'क', 'ख', 'क्ष', 'ल', 'लृ', 'luu',
    'म', 'न,', '९', 'ण', 'न', 'ञ', 'ओ', '१', 'प', 'फ',
    'र', 'ऋ', 'rii', 'ष', 'स', '७', 'श', '६', 'ट', 'ठ',
    '३', 'त्र', 'त', 'थ', '२', 'उ', 'ऊ', 'व', 'य', '०',
]

IMAGE_SIZE = 64  # Model input size
MLFLOW_PORT = 5002  # Port for MLflow UI

def prepare_image(image, model_name):
    """Preprocess the image based on the selected model."""
    image = Image.open(image).convert("RGB")  # Ensure RGB mode
    image = np.array(image, dtype=np.uint8)

    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    _, otsu_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    target_size = (64, 64) if model_name == 'ran2dev' else (32, 32)
    otsu_image = cv2.resize(otsu_image, target_size, interpolation=cv2.INTER_AREA)
    otsu_image = otsu_image / 255.0
    otsu_image = np.reshape(otsu_image, (1, target_size[1], target_size[0], 1))
    return otsu_image

def segment_characters(image, model_name):
    """Segment characters from an image and preprocess them according to the model."""
    image = Image.open(image).convert("L")
    image = np.array(image, dtype=np.uint8)
    
    _, thresh = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    character_images = []
    bounding_boxes = []
    target_size = (64, 64) if model_name == 'ran2dev' else (32, 32)
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        char_image = image[y:y+h, x:x+w]
        char_image = cv2.resize(char_image, target_size)
        char_image = char_image.astype('float32') / 255.0
        char_image = np.expand_dims(char_image, axis=(0, -1))
        character_images.append(char_image)
        bounding_boxes.append((x, y, w, h))
    
    sorted_characters = sorted(zip(character_images, bounding_boxes), key=lambda b: b[1][0])
    return [char[0] for char in sorted_characters]


def predict_characters(image, model_name):
    """Predict characters using the chosen model."""
    character_images = segment_characters(image, model_name)
    model = models[model_name]
    combined_prediction = ""

    for char_image in character_images:
        pred = model.predict(char_image)
        predicted_class_index = np.argmax(pred, axis=-1)[0]
        predicted_class_label = class_labels[predicted_class_index]
        combined_prediction += predicted_class_label

    return combined_prediction

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/learn')
def learn():
    return render_template('learn.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files or 'model' not in request.form:
        return jsonify({"error": "No file or model selected"})
    
    image_file = request.files['image']
    model_name = request.form['model']
    
    if image_file.filename == '':
        return jsonify({"error": "No selected file"})
    
    if model_name not in models:
        return jsonify({"error": "Invalid model selected"})
    
    try:
        image = prepare_image(image_file, model_name)
        model = models[model_name]
        prediction = model.predict(image)
        predicted_class = np.argmax(prediction, axis=1)[0]
        predicted_label = class_labels[predicted_class]
        
        return jsonify({"prediction": predicted_label})
    except Exception as e:
        return jsonify({"error": f"An error occurred: {e}"})

@app.route('/predict-word', methods=['POST'])
def predict_word():
    if 'image' not in request.files or 'model' not in request.form:
        return jsonify({"error": "No file or model selected"})
    
    image_file = request.files['image']
    model_name = request.form['model']
    
    if image_file.filename == '' or model_name not in models:
        return jsonify({"error": "Invalid input"})
    
    try:
        predicted_text = predict_characters(image_file, model_name)
        return jsonify({"prediction": predicted_text})
    except Exception as e:
        return jsonify({"error": f"An error occurred: {e}"})

@app.route('/compare_models')
def compare_models():
    """Launch MLflow UI and redirect to it."""
    try:
        subprocess.Popen(
            f"mlflow ui --backend-store-uri file:///C:/Users/Admin/mlruns --port {MLFLOW_PORT}",
            shell=True
        )
        return redirect(f"http://localhost:{MLFLOW_PORT}")
    except Exception as e:
        print(f"Error opening MLflow UI: {e}")
        return "Error opening MLflow UI."

if __name__ == '__main__':
    app.run(debug=True)
