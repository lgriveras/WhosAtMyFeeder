import os
import time
import yaml
import sys
import numpy as np
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from PIL import Image, ImageOps
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
from datetime import datetime
from queries import get_common_name
import argparse
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("file_watcher.log")
    ]
)
logger = logging.getLogger("BirdWatcher")

# Global variables
classifier = None
config = None

# Image extensions to monitor
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}
PROCESSED_SUFFIX = "_processed"

def load_config():
    """Load configuration from config.yml file"""
    global config
    file_path = './config/config.yml'
    with open(file_path, 'r') as config_file:
        config = yaml.safe_load(config_file)
    
    # Add default file_watcher config if not present
    if 'file_watcher' not in config:
        config['file_watcher'] = {
            'watch_directory': './watch',
            'threshold': 0.7,
            'use_common_names': True,
            'rename_files': True
        }
    
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
    logger.info("Model initialized successfully")
    return classifier

def preprocess_image(image_path):
    """Preprocess image for classification"""
    # Open the image from file
    image = Image.open(image_path)
    
    # Resize the image while maintaining its aspect ratio
    max_size = (224, 224)
    image.thumbnail(max_size)
    
    # Pad the image to fill the remaining space to ensure it's exactly 224x224
    padded_image = ImageOps.expand(image, border=(
        (max_size[0] - image.size[0]) // 2,
        (max_size[1] - image.size[1]) // 2
    ), fill='black')
    
    return padded_image

def classify(image):
    """Classify image using the TensorFlow Lite model"""
    tensor_image = vision.TensorImage.create_from_array(image)
    categories = classifier.classify(tensor_image)
    return categories.classifications[0].categories

def process_image(image_path):
    """Process image and return top classification results"""
    try:
        logger.info(f"Processing image: {image_path}")
        
        # Preprocess the image
        processed_image = preprocess_image(image_path)
        
        # Convert to numpy array
        np_arr = np.array(processed_image)
        
        # Run classification
        categories = classify(np_arr)
        
        # Process results
        results = []
        for category in categories:
            common_name = get_common_name(category.display_name)
            
            results.append({
                'index': int(category.index),
                'score': float(category.score),
                'scientific_name': category.display_name,
                'category_name': category.category_name,
                'common_name': common_name
            })
        
        # Filter out background category (index 964) and apply threshold
        threshold = config['file_watcher'].get('threshold', config['classification'].get('threshold', 0.7))
        filtered_results = [r for r in results if r['index'] != 964 and r['score'] >= threshold]
        
        if filtered_results:
            top_result = filtered_results[0]
            logger.info(f"Identified: {top_result['scientific_name']} ({top_result['common_name']}) with score {top_result['score']:.4f}")
            return top_result
        else:
            logger.info("No confident identification found")
            return None
        
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {str(e)}")
        return None

def rename_image_file(image_path, identification):
    """Rename image file to include bird species information"""
    if not config['file_watcher'].get('rename_files', True):
        return image_path
    
    try:
        path = Path(image_path)
        
        # Check if the file has already been processed
        if PROCESSED_SUFFIX in path.stem:
            logger.info(f"File already processed: {path.name}")
            return str(path)
        
        # Determine the species name to use (common or scientific)
        if config['file_watcher'].get('use_common_names', True) and identification['common_name'] != "No common name found.":
            species_name = identification['common_name']
        else:
            species_name = identification['scientific_name']
            
        # Clean up the species name for use in a filename
        species_name = species_name.replace(" ", "_").replace("/", "-").replace("\\", "-")
        
        # Create new filename with species info and score
        score_str = f"{identification['score']:.2f}".replace(".", "p")
        new_filename = f"{path.stem}{PROCESSED_SUFFIX}_{species_name}_{score_str}{path.suffix}"
        new_path = path.parent / new_filename
        
        # Rename the file
        os.rename(str(path), str(new_path))
        logger.info(f"Renamed: {path.name} to {new_path.name}")
        return str(new_path)
    
    except Exception as e:
        logger.error(f"Error renaming file {image_path}: {str(e)}")
        return image_path

class BirdImageHandler(FileSystemEventHandler):
    """Handler for file system events"""
    
    def on_created(self, event):
        """Handle file creation event"""
        logger.info(f"File created: {event.src_path}")

        if event.is_directory:
            return
            
        file_path = event.src_path
        file_ext = os.path.splitext(file_path)[1].lower()
        
        # Check if the file is an image and not already processed
        if (file_ext in IMAGE_EXTENSIONS and 
            PROCESSED_SUFFIX not in os.path.basename(file_path)):
            
            # Wait a moment to ensure the file is completely written
            time.sleep(1)
            
            # Process the image
            identification = process_image(file_path)
            
            # Rename the file if identification was successful
            if identification:
                rename_image_file(file_path, identification)

def start_watching(directory):
    """Start watching the directory for new image files"""
    directory = os.path.abspath(directory)  # Get absolute path
    logger.info(f"Starting bird image watcher on directory: {directory}")
    
    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created watch directory: {directory}")
    
    # Create the observer and start watching
    event_handler = BirdImageHandler()
    observer = Observer()
    
    # For Docker on Windows, use polling observer which is more reliable
    # but less efficient
    try:
        observer.schedule(event_handler, directory, recursive=True)
        observer.start()
        logger.info(f"Observer started for: {directory}")
    except Exception as e:
        logger.error(f"Error starting observer: {str(e)}")
        raise
    
    try:
        logger.info("Bird watcher running. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopping bird watcher...")
        observer.stop()
    
    observer.join()

def process_existing_files(directory):
    """Process existing files in the directory"""
    logger.info(f"Processing existing files in: {directory}")
    
    for root, _, files in os.walk(directory):
        for filename in files:
            file_path = os.path.join(root, filename)
            file_ext = os.path.splitext(filename)[1].lower()
            
            if (file_ext in IMAGE_EXTENSIONS and 
                PROCESSED_SUFFIX not in filename):
                
                identification = process_image(file_path)
                
                if identification:
                    rename_image_file(file_path, identification)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Bird Species Image Watcher')
    parser.add_argument('--directory', '-d', type=str, help='Directory to watch for new images')
    parser.add_argument('--process-existing', '-p', action='store_true', help='Process existing files in the directory')
    parser.add_argument('--threshold', '-t', type=float, help='Confidence threshold for identification (0.0-1.0)')
    parser.add_argument('--use-common-names', '-c', action='store_true', help='Use common names instead of scientific names')
    
    args = parser.parse_args()
    
    # Load config
    global config
    config = load_config()
    
    # Override config with command-line arguments if provided
    if args.directory:
        config['file_watcher']['watch_directory'] = args.directory
    if args.threshold:
        config['file_watcher']['threshold'] = args.threshold
    if args.use_common_names:
        config['file_watcher']['use_common_names'] = True
    
    # Initialize the model
    initialize_model()
    
    watch_directory = config['file_watcher']['watch_directory']
    
    # Process existing files if requested
    if args.process_existing:
        process_existing_files(watch_directory)
    
    # Start watching for new files
    start_watching(watch_directory)

if __name__ == '__main__':
    main()