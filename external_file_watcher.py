import os
import time
import yaml
import sys
import requests
import json
import base64
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from PIL import Image
from datetime import datetime
import argparse
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("external_file_watcher.log")
    ]
)
logger = logging.getLogger("ExternalBirdWatcher")

# Global variables
config = None

# Image extensions to monitor
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}
PROCESSED_SUFFIX = "_processed"

# API configuration
API_URL = "http://localhost:8000"

def load_config():
    """Load configuration from config.yml file"""
    global config
    file_path = './config/config.yml'
    
    # Use default config if file doesn't exist
    if not os.path.exists(file_path):
        logger.warning(f"Config file not found at {file_path}. Using default configuration.")
        config = {
            'external_file_watcher': {
                'watch_directory': './watch',
                'threshold': 0.7,
                'use_common_names': True,
                'rename_files': True,
                'api_url': API_URL
            }
        }
        return config
    
    with open(file_path, 'r') as config_file:
        config = yaml.safe_load(config_file)
    
    # Add default external_file_watcher config if not present
    if 'external_file_watcher' not in config:
        config['external_file_watcher'] = {
            'watch_directory': './watch',
            'threshold': 0.7,
            'use_common_names': True,
            'rename_files': True,
            'api_url': API_URL
        }
    
    return config

def encode_image_to_base64(image_path):
    """Encode image file to base64 for API submission"""
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return encoded_string
    except Exception as e:
        logger.error(f"Error encoding image {image_path}: {str(e)}")
        return None

def process_image(image_path):
    """Process image by sending it to the API and return classification results"""
    try:
        logger.info(f"Processing image: {image_path}")
        
        # Encode image to base64
        encoded_image = encode_image_to_base64(image_path)
        if not encoded_image:
            return None
        
        # Prepare data for API
        api_url = config['external_file_watcher'].get('api_url', API_URL)
        endpoint = f"{api_url}/api/identify"
        
        # Send the request to the API using form data (multipart/form-data)
        files = {
            "image": (os.path.basename(image_path), open(image_path, "rb"), "image/jpeg")
        }
        
        # Send the request to the API
        response = requests.post(endpoint, files=files)
        
        # Check if the request was successful
        if response.status_code == 200:
            result = response.json()
            logger.info(f"API successfully processed the image {result}")
            
            if result and "status" in result and result["status"] == "success" and "results" in result:
                if len(result["results"]) > 0:
                    bird_data = result["results"][0]
                    identification = {
                    "scientific_name": bird_data["display_name"],
                    "common_name": bird_data["common_name"] if "common_name" in bird_data else "No common name found.",
                    "score": bird_data["score"],
                    "category": bird_data["category_name"]
                    }
                    logger.info(f"Identified: {identification['scientific_name']} "
                        f"({identification['common_name']}) with score {identification['score']:.4f}")
                    return identification
                else:
                    logger.info("No identification results returned by API")
                    return None
            else:
                logger.warning(f"Unexpected API response format: {result}")
            return None
        else:
            logger.error(f"API request failed with status code {response.status_code}: {response.text}")
            return None
            
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection error to API: {str(e)}. Is the API running at {api_url}?")
        return None
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {str(e)}")
        return None

def rename_image_file(image_path, identification):
    """Rename image file to include bird species information"""
    if not config['external_file_watcher'].get('rename_files', True):
        return image_path
    
    try:
        path = Path(image_path)
        
        # Check if the file has already been processed
        if PROCESSED_SUFFIX in path.stem:
            logger.info(f"File already processed: {path.name}")
            return str(path)
        
        # Determine the species name to use (common or scientific)
        if config['external_file_watcher'].get('use_common_names', True) and identification['common_name'] != "No common name found.":
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

class ExternalBirdImageHandler(FileSystemEventHandler):
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
            
            # Process the image via API
            identification = process_image(file_path)
            
            # Rename the file if identification was successful
            if identification:
                rename_image_file(file_path, identification)

def start_watching(directory):
    """Start watching the directory for new image files"""
    directory = os.path.abspath(directory)  # Get absolute path
    logger.info(f"Starting external bird image watcher on directory: {directory}")
    
    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created watch directory: {directory}")
    
    # Create the observer and start watching
    event_handler = ExternalBirdImageHandler()
    observer = Observer()
    
    try:
        observer.schedule(event_handler, directory, recursive=True)
        observer.start()
        logger.info(f"Observer started for: {directory}")
    except Exception as e:
        logger.error(f"Error starting observer: {str(e)}")
        raise
    
    # Check if API is available
    try:
        api_url = config['external_file_watcher'].get('api_url', API_URL)
        health_check_url = f"{api_url}/api/health"
        response = requests.get(health_check_url)
        if response.status_code == 200:
            logger.info(f"API is available at {api_url}")
        else:
            logger.warning(f"API health check failed with status code {response.status_code}")
    except requests.exceptions.ConnectionError:
        logger.warning(f"Could not connect to API at {api_url}. Make sure the API is running.")
    
    try:
        logger.info("External bird watcher running. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopping external bird watcher...")
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
    parser = argparse.ArgumentParser(description='External Bird Species Image Watcher (Using API)')
    parser.add_argument('--directory', '-d', type=str, help='Directory to watch for new images')
    parser.add_argument('--process-existing', '-p', action='store_true', help='Process existing files in the directory')
    parser.add_argument('--threshold', '-t', type=float, help='Confidence threshold for identification (0.0-1.0)')
    parser.add_argument('--use-common-names', '-c', action='store_true', help='Use common names instead of scientific names')
    parser.add_argument('--api-url', '-a', type=str, help='URL of the API (default: http://localhost:8000)')
    
    args = parser.parse_args()
    
    # Load config
    global config
    config = load_config()
    
    # Override config with command-line arguments if provided
    if args.directory:
        config['external_file_watcher']['watch_directory'] = args.directory
    if args.threshold:
        config['external_file_watcher']['threshold'] = args.threshold
    if args.use_common_names:
        config['external_file_watcher']['use_common_names'] = True
    if args.api_url:
        config['external_file_watcher']['api_url'] = args.api_url
    
    watch_directory = config['external_file_watcher']['watch_directory']
    
    # Process existing files if requested
    if args.process_existing:
        process_existing_files(watch_directory)
    
    # Start watching for new files
    start_watching(watch_directory)

if __name__ == '__main__':
    main()