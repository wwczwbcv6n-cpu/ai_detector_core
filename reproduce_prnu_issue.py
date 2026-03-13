
import sys
import numpy as np
from PIL import Image
import os

# Add src to path
sys.path.append('src')

from prnu_features import extract_prnu_features

def test_prnu():
    # Create a dummy image
    img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    print("Testing extract_prnu_features...")
    features = extract_prnu_features(img)
    print(f"Features: {features}")
    
    if np.all(features == 0):
        print("Error: Features are all zero. Extraction failed.")
    else:
        print("Success: Features extracted.")

if __name__ == "__main__":
    test_prnu()
