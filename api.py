import sqlite3
import numpy as np
from datetime import datetime
import yaml
import sys
import json
from flask import Flask, request, jsonify
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
from PIL import Image, ImageOps
from io import BytesIO
from queries import get_common_name

app = Flask(__name__)

# Global variables
classifier = None
config = None
DBPATH = './data/speciesid.db'

def load_config():
    """Load configuration from config.yml file"""
    global config
    file_path = './config/config.yml'
    with open(file_path, 'r') as config_file:
        config = yaml.safe_load(config_file)
    return config

def initialize_model():
    """Initialize the TensorFlow Lite model for bird species classification"""
    global classifier
    
    # Initialize the image classification model
    base_options = core.BaseOptions(
        file_name=config['classification']['model'], use_coral=False, num_threads=4)

    # Configure classification options
    classification_options = processor.ClassificationOptions(
        max_results=5, score_threshold=0)  # Return top 5 results
    options = vision.ImageClassifierOptions(
        base_options=base_options, classification_options=classification_options)

    # Create classifier
    classifier = vision.ImageClassifier.create_from_options(options)
    return classifier

def classify(image):
    """Classify image using the TensorFlow Lite model"""
    tensor_image = vision.TensorImage.create_from_array(image)
    categories = classifier.classify(tensor_image)
    return categories.classifications[0].categories

def preprocess_image(image_data):
    """Preprocess image for classification"""
    # Open the image from bytes data
    image = Image.open(BytesIO(image_data))
    
    # Resize the image while maintaining its aspect ratio
    max_size = (224, 224)
    image.thumbnail(max_size)
    
    # Pad the image to fill the remaining space to ensure it's exactly 224x224
    padded_image = ImageOps.expand(image, border=(
        (max_size[0] - image.size[0]) // 2,
        (max_size[1] - image.size[1]) // 2
    ), fill='black')
    
    return padded_image

@app.route('/api/identify', methods=['POST'])
def identify_bird():
    """API endpoint to identify bird species from an uploaded image"""
    # Check if image file is present in the request
    if 'image' not in request.files:
        return jsonify({
            'error': 'No image provided',
            'status': 'error'
        }), 400
    
    image_file = request.files['image']
    
    # Read the image data
    image_data = image_file.read()
    
    # Preprocess the image
    try:
        processed_image = preprocess_image(image_data)
        
        # Convert to numpy array
        np_arr = np.array(processed_image)
        
        # Run classification
        categories = classify(np_arr)
        
        # Process results
        results = []
        for category in categories:
            # Add common name if available
            common_name = get_common_name(category.display_name)
            
            results.append({
                'index': int(category.index),
                'score': float(category.score),
                'display_name': category.display_name,  # Scientific name
                'category_name': category.category_name,
                'common_name': common_name
            })
        
        # Filter out background category (index 964) and apply threshold
        threshold = config['classification'].get('threshold', 0.5)
        filtered_results = [r for r in results if r['index'] != 964 and r['score'] >= threshold]
        
        return jsonify({
            'status': 'success',
            'results': filtered_results,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """API endpoint for health check"""
    return jsonify({
        'status': 'healthy',
        'model': config['classification']['model'],
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

def main():
    """Initialize and run the API server"""
    global config
    
    print("Starting Bird Species Identification API", flush=True)
    print("Python version:", sys.version, flush=True)
    
    # Load configuration
    config = load_config()
    
    # Initialize model
    initialize_model()
    
    # Run the Flask app
    port = config.get('api', {}).get('port', 5000)
    host = config.get('api', {}).get('host', '0.0.0.0')
    debug = config.get('api', {}).get('debug', False)
    
    print(f"API server running on {host}:{port}", flush=True)
    app.run(debug=debug, host=host, port=port)

if __name__ == '__main__':
    main()