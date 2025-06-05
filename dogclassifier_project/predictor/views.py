# views.py
import os
import numpy as np
import csv
import base64
from io import BytesIO
from django.shortcuts import render
from django.conf import settings
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load class labels from CSV (only 'breed' column)
def load_class_names(csv_path):
    class_names = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            class_names.append(row['breed'])
    return class_names

# Paths
model_path = os.path.join(settings.BASE_DIR, 'predictor', 'model', 'best_dog_breed_model.keras')
labels_path = os.path.join(settings.BASE_DIR, 'predictor', 'model', 'labels.csv')

# Load model and class names once
model = load_model(model_path)
CLASS_NAMES = load_class_names(labels_path)

# Preprocess image to match Gradio pipeline
def preprocess_image(img):
    img = img.convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Get top 5 predictions with labels and confidences
def get_top_predictions(predictions, class_names, threshold=0.6):
    top_indices = predictions[0].argsort()[-5:][::-1]
    top_labels = [(class_names[i], predictions[0][i] * 100) for i in top_indices]
    final_label, final_conf = top_labels[0]
    if final_conf < (threshold * 100):
        final_label = "Unknown"
    return final_label, top_labels

# Main view
def index(request):
    if request.method == 'POST':
        img = None

        # Handle uploaded file
        if 'image' in request.FILES:
            img_file = request.FILES['image']
            img = Image.open(img_file)

        # Handle webcam image (base64)
        elif 'webcam_image' in request.POST and request.POST['webcam_image']:
            data_url = request.POST['webcam_image']
            header, encoded = data_url.split(',', 1)
            img_data = base64.b64decode(encoded)
            img = Image.open(BytesIO(img_data))

        if img:
            processed_image = preprocess_image(img)
            prediction = model.predict(processed_image)
            predicted_class, top_predictions = get_top_predictions(prediction, CLASS_NAMES)

            return render(request, 'predictor/index.html', {
                'predicted_class': predicted_class,
                'top_predictions': top_predictions
            })

    return render(request, 'predictor/index.html')
