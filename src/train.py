import os
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
from PIL import Image

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    _TF_AVAILABLE = True
except ImportError:
    tf = None
    _TF_AVAILABLE = False
    print("[train.py] TensorFlow not available — TF-based training disabled.")

try:
    import pyheif
    _PYHEIF_AVAILABLE = True
except ImportError:
    try:
        import pillow_heif as _pillow_heif
        _pillow_heif.register_heif_opener()
        _PYHEIF_AVAILABLE = False   # use pillow path instead
    except ImportError:
        _PYHEIF_AVAILABLE = False
    pyheif = None

try:
    from moviepy.video.io.VideoFileClip import VideoFileClip
    _MOVIEPY_AVAILABLE = True
except ImportError:
    VideoFileClip = None
    _MOVIEPY_AVAILABLE = False
    print("[train.py] moviepy not available — video loading disabled.")

# --- Configuration ---
IMG_WIDTH = 256
IMG_HEIGHT = 256
BATCH_SIZE = 32
FRAMES_PER_VIDEO = 5 # Number of frames to extract from each video

# Get the absolute path of the script's directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Construct absolute paths for data and models directories
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
MODELS_DIR = os.path.join(SCRIPT_DIR, '..', 'models')
GENERATED_AI_IMAGES_DIR = os.path.join(SCRIPT_DIR, '..', 'data', 'generated_ai', 'images')
EDITED_AI_IMAGES_DIR = os.path.join(SCRIPT_DIR, '..', 'data', 'edited_ai', 'images')

def load_heic(path):
    """Loads a HEIC image and converts it to a CV2 format."""
    try:
        heif_file = pyheif.read(path)
        image = Image.frombytes(
            heif_file.mode, 
            heif_file.size, 
            heif_file.data,
            "raw",
            heif_file.mode,
            heif_file.stride,
        )
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"Error loading HEIC file {path}: {e}")
        return None

def extract_frames_from_video(video_path, num_frames=FRAMES_PER_VIDEO):
    """
    Extracts a fixed number of frames from a video file.
    """
    frames = []
    try:
        clip = VideoFileClip(video_path)
        
        # Calculate time steps to extract frames evenly
        duration = clip.duration
        if duration == 0:
            print(f"Warning: Video {video_path} has zero duration.")
            return frames

        # Ensure we don't try to extract more frames than available or from a very short video
        actual_frames_to_extract = min(num_frames, int(duration * clip.fps))
        
        # Extract frames
        for i in range(actual_frames_to_extract):
            # Get frame at even intervals
            time_in_seconds = (i / (actual_frames_to_extract - 1 if actual_frames_to_extract > 1 else 1)) * duration
            frame = clip.get_frame(time_in_seconds) # frame is a numpy array (H, W, 3)
            frames.append(frame)
        
        clip.close()
    except Exception as e:
        print(f"Error extracting frames from video {video_path}: {e}")
    return frames

def load_data():
    """
    Loads and preprocesses image and video data from various directories.
    
    Returns:
        A tuple of (images, labels) as numpy arrays.
    """
    print(f"Loading data...")
    images = []
    labels = []
    
    # --- Load from original data/real and data/ai ---
    for label, category in enumerate(['real', 'ai']):
        category_path = os.path.join(DATA_DIR, category)
        if not os.path.exists(category_path):
            print(f"Warning: Directory not found: {category_path}")
            continue
            
        print(f"Loading {category} content from {category_path}")
        for filename in os.listdir(category_path):
            file_path = os.path.join(category_path, filename)
            
            ext = os.path.splitext(filename)[1].lower()
            
            if ext in ['.jpg', '.jpeg', '.png', '.heic']:
                img = None
                if ext == '.heic':
                    img = load_heic(file_path)
                else:
                    img = cv2.imread(file_path)

                if img is not None:
                    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                    img = img / 255.0 # Normalize to [0, 1]
                    images.append(img)
                    labels.append(label)
                    # print(f"Loaded and processed: {filename} (Label: {category})")
                else:
                    print(f"Warning: Could not load image {filename}")
            elif ext in ['.mp4', '.mov']:
                print(f"Extracting frames from video: {filename} (Label: {category})")
                video_frames = extract_frames_from_video(file_path)
                for frame in video_frames:
                    frame_img = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
                    frame_img = frame_img / 255.0 # Normalize to [0, 1]
                    images.append(frame_img)
                    labels.append(label)
            else:
                print(f"Skipping unsupported file type: {filename} in {category_path}")

    # --- Load from generated_ai/images (always label 1 for AI) ---
    if os.path.exists(GENERATED_AI_IMAGES_DIR):
        print(f"Loading AI-generated images from {GENERATED_AI_IMAGES_DIR}")
        for filename in os.listdir(GENERATED_AI_IMAGES_DIR):
            file_path = os.path.join(GENERATED_AI_IMAGES_DIR, filename)
            ext = os.path.splitext(filename)[1].lower()
            if ext in ['.jpg', '.jpeg', '.png', '.heic']:
                img = None
                if ext == '.heic':
                    img = load_heic(file_path)
                else:
                    img = cv2.imread(file_path)

                if img is not None:
                    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                    img = img / 255.0 # Normalize to [0, 1]
                    images.append(img)
                    labels.append(1) # Label as AI
                    # print(f"Loaded and processed: {filename} (Label: AI)")
                else:
                    print(f"Warning: Could not load image {filename} from generated AI dir")
            else:
                print(f"Skipping unsupported file type: {filename} in {GENERATED_AI_IMAGES_DIR}")
    else:
        print(f"Warning: Directory not found: {GENERATED_AI_IMAGES_DIR}")

    # --- Load from edited_ai/images (always label 1 for AI) ---
    if os.path.exists(EDITED_AI_IMAGES_DIR):
        print(f"Loading AI-edited images from {EDITED_AI_IMAGES_DIR}")
        for filename in os.listdir(EDITED_AI_IMAGES_DIR):
            file_path = os.path.join(EDITED_AI_IMAGES_DIR, filename)
            ext = os.path.splitext(filename)[1].lower()
            if ext in ['.jpg', '.jpeg', '.png', '.heic']:
                img = None
                if ext == '.heic':
                    img = load_heic(file_path)
                else:
                    img = cv2.imread(file_path)

                if img is not None:
                    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                    img = img / 255.0 # Normalize to [0, 1]
                    images.append(img)
                    labels.append(1) # Label as AI
                    # print(f"Loaded and processed: {filename} (Label: AI)")
                else:
                    print(f"Warning: Could not load image {filename} from edited AI dir")
            else:
                print(f"Skipping unsupported file type: {filename} in {EDITED_AI_IMAGES_DIR}")
    else:
        print(f"Warning: Directory not found: {EDITED_AI_IMAGES_DIR}")


    if not images:
        print("Error: No images were loaded. Please check the data directories and file formats.")
        return None, None
        
    return np.array(images), np.array(labels)

def build_model(input_shape):
    """
    Builds a simple Convolutional Neural Network (CNN) model.
    
    Args:
        input_shape (tuple): The shape of the input images.
        
    Returns:
        A TensorFlow Keras model.
    """
    print("Building the CNN model...")
    
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid')) # Sigmoid for binary classification
    
    model.summary()
    
    return model

def train_model(model, images, labels):
    """
    Trains the model on the given images and labels.
    
    Args:
        model: The TensorFlow Keras model to train.
        images: The list of images.
        labels: The list of labels.
    """
    if images is None or labels is None or len(images) == 0:
        print("No data to train on. Please check the data loading step.")
        return
        
    print("\n--- Starting Model Training ---")
    
    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        images, labels, test_size=0.5, random_state=42, stratify=labels
    )
    
    print(f"Training with {len(X_train)} images, validating with {len(X_val)} images.")
    
    # Compile the model
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
                  
    # Train the model
    # Note: With a very small dataset, this will be quick and not very accurate.
    history = model.fit(X_train, y_train, 
                        epochs=10, 
                        batch_size=BATCH_SIZE, 
                        validation_data=(X_val, y_val))
                        
    print("\n--- Model Training Finished ---")
    
    # Save the trained model
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
    model_path = os.path.join(MODELS_DIR, 'ai_detector_model.h5')
    model.save(model_path)
    print(f"Model saved to: {model_path}")

def main():
    """
    The main function to run the training pipeline.
    """
    print("--- AI Detector Model Training Pipeline ---")
    
    # 1. Load the data
    images, labels = load_data()
    
    # If data loading fails, exit.
    if images is None or labels is None:
        return
        
    # 2. Build the model
    input_shape = (IMG_HEIGHT, IMG_WIDTH, 3) 
    model = build_model(input_shape)
    
    # 3. Train the model
    train_model(model, images, labels)
    
    print("\n--- Pipeline execution finished. ---")

if __name__ == '__main__':
    main()