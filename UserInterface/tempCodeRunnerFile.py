from flask import Flask, render_template, request, jsonify
from keras.models import load_model
import numpy as np
import cv2
from PIL import Image
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__)

# Load models once at the beginning
models = {
    'lenet5': load_model('50lenet5.keras'),
    'ran2dev': load_model('50ran2dev.keras')
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

def prepare_image(image, model_name):
    """Preprocess the image: Resize, convert to grayscale, and apply Otsu thresholding."""
    image = Image.open(image).convert("RGB")  # Ensure RGB mode
    image = np.array(image, dtype=np.uint8)  # Convert to 8-bit unsigned integer

    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    image = cv2.resize(image, (67, 78), interpolation=cv2.INTER_AREA)
    _, otsu_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    target_size = (64, 64) if model_name == 'ran2dev' else (32, 32)
    otsu_image = cv2.resize(otsu_image, target_size, interpolation=cv2.INTER_AREA)
    otsu_image = otsu_image / 255.0
    otsu_image = np.reshape(otsu_image, (1, target_size[1], target_size[0], 1))

    return otsu_image

def segment_characters(image):
    image = Image.open(image).convert("L")
    image = np.array(image, dtype=np.uint8)
    
    _, thresh = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    character_images = []
    bounding_boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        char_image = image[y:y+h, x:x+w]
        char_image = cv2.resize(char_image, (IMAGE_SIZE, IMAGE_SIZE))
        char_image = char_image.astype('float32') / 255.0
        char_image = np.expand_dims(char_image, axis=(0, -1))
        character_images.append(char_image)
        bounding_boxes.append((x, y, w, h))
    
    sorted_characters = sorted(zip(character_images, bounding_boxes), key=lambda b: b[1][0])
    return [char[0] for char in sorted_characters]

def predict_characters(image):
    character_images = segment_characters(image)
    combined_prediction = ""

    for char_image in character_images:
        pred = models['ran2dev'].predict(char_image)
        predicted_class_index = np.argmax(pred, axis=-1)[0]
        predicted_class_label = class_labels[predicted_class_index]
        combined_prediction += predicted_class_label

    return combined_prediction


@app.route('/')
def index():
    return render_template('index.html')

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
        confidence = np.max(prediction) * 100
        predicted_label = class_labels[predicted_class]
        
        return jsonify({
            "prediction": predicted_label,
            # "confidence": f"{confidence:.2f}%"
        })
    except Exception as e:
        return jsonify({"error": f"An error occurred: {e}"})

@app.route('/predict-word', methods=['POST'])
def predict_word():
    if 'image' not in request.files:
        return jsonify({"error": "No file provided"})
    
    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({"error": "No selected file"})
    
    try:
        predicted_text = predict_characters(image_file)

        return jsonify({
            "prediction": predicted_text,
            # "confidence": f"{word_confidence:.2f}%
        })
    except Exception as e:
        return jsonify({"error": f"An error occurred: {e}"})
    
if __name__ == '__main__':
    app.run(debug=True)
