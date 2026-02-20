import os
import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import random

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_REAL_DIR = os.path.join(SCRIPT_DIR, '..', 'data', 'real')
INPUT_GENERATED_AI_DIR = os.path.join(SCRIPT_DIR, '..', 'data', 'generated_ai', 'images')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, '..', 'data', 'edited_ai', 'images')

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def apply_blur(image_path, output_path, blur_radius=2):
    """Applies a Gaussian blur to an image."""
    try:
        img = Image.open(image_path)
        blurred_img = img.filter(ImageFilter.GaussianBlur(blur_radius))
        blurred_img.save(output_path)
        print(f"Applied blur to {os.path.basename(image_path)} and saved to {output_path}")
    except Exception as e:
        print(f"Error applying blur to {image_path}: {e}")

def apply_color_balance(image_path, output_path, red_factor=1.1, green_factor=1.0, blue_factor=0.9):
    """Adjusts color balance of an image."""
    try:
        img = Image.open(image_path).convert("RGB")
        data = np.array(img)
        
        # Split into R, G, B channels
        r, g, b = data[:,:,0], data[:,:,1], data[:,:,2]
        
        # Apply factors
        r = np.clip(r * red_factor, 0, 255).astype(np.uint8)
        g = np.clip(g * green_factor, 0, 255).astype(np.uint8)
        b = np.clip(b * blue_factor, 0, 255).astype(np.uint8)
        
        # Recombine and save
        edited_data = np.stack([r, g, b], axis=-1)
        edited_img = Image.fromarray(edited_data)
        edited_img.save(output_path)
        print(f"Applied color balance to {os.path.basename(image_path)} and saved to {output_path}")
    except Exception as e:
        print(f"Error applying color balance to {image_path}: {e}")

def overlay_text(image_path, output_path, text="AI Edited", position=(10, 10), font_size=30, color=(255, 255, 255)):
    """Overlays text on an image."""
    try:
        from PIL import ImageDraw, ImageFont
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        
        # Try to load a default font, or use a generic one
        try:
            font = ImageFont.truetype("arial.ttf", font_size) # Common font on Linux/Windows
        except IOError:
            font = ImageFont.load_default() # Fallback
            print("Could not load 'arial.ttf', using default font.")

        draw.text(position, text, font=font, fill=color)
        img.save(output_path)
        print(f"Overlaid text on {os.path.basename(image_path)} and saved to {output_path}")
    except Exception as e:
        print(f"Error overlaying text on {image_path}: {e}")

def add_noise(image_path, output_path, noise_factor=0.1):
    """Adds Gaussian noise to an image."""
    try:
        img = Image.open(image_path).convert("RGB")
        img_np = np.array(img)
        
        noise = np.random.normal(0, noise_factor * 255, img_np.shape)
        noisy_img_np = np.clip(img_np + noise, 0, 255).astype(np.uint8)
        noisy_img = Image.fromarray(noisy_img_np)
        noisy_img.save(output_path)
        print(f"Added noise to {os.path.basename(image_path)} and saved to {output_path}")
    except Exception as e:
        print(f"Error adding noise to {image_path}: {e}")

def apply_compression(image_path, output_path, quality=50):
    """Applies JPEG compression to an image."""
    try:
        img = Image.open(image_path).convert("RGB")
        img.save(output_path, quality=quality, optimize=True)
        print(f"Applied JPEG compression (quality={quality}) to {os.path.basename(image_path)} and saved to {output_path}")
    except Exception as e:
        print(f"Error applying compression to {image_path}: {e}")

def apply_rescale(image_path, output_path, scale_factor=0.5):
    """Rescales an image down and then up, introducing artifacts."""
    try:
        img = Image.open(image_path).convert("RGB")
        original_size = img.size
        
        # Rescale down
        new_size_down = (int(original_size[0] * scale_factor), int(original_size[1] * scale_factor))
        img_down = img.resize(new_size_down, Image.BICUBIC) # Using BICUBIC for downscaling

        # Rescale up
        img_up = img_down.resize(original_size, Image.BICUBIC)
        img_up.save(output_path)
        print(f"Applied rescale (down by {scale_factor}) to {os.path.basename(image_path)} and saved to {output_path}")
    except Exception as e:
        print(f"Error applying rescale to {image_path}: {e}")

def apply_color_jitter(image_path, output_path, brightness_factor=1.2, contrast_factor=0.8, saturation_factor=1.5):
    """Applies random color jitter to an image."""
    try:
        img = Image.open(image_path).convert("RGB")
        
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(brightness_factor)
        
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(contrast_factor)
        
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(saturation_factor)
        
        img.save(output_path)
        print(f"Applied color jitter to {os.path.basename(image_path)} and saved to {output_path}")
    except Exception as e:
        print(f"Error applying color jitter to {image_path}: {e}")


def main():
    """Main function to run image editing."""
    print("--- AI Image Editing Script ---")

    # Get a real image to edit
    real_images = [os.path.join(INPUT_REAL_DIR, f) for f in os.listdir(INPUT_REAL_DIR) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.heic'))]
    if not real_images:
        print(f"No real images found in {INPUT_REAL_DIR} to edit. Skipping real image edits.")
    else:
        # Prioritize JPG/PNG to avoid HEIC conversion issues if not needed by PIL directly
        real_image_path = next((f for f in real_images if f.lower().endswith(('.jpg', '.jpeg', '.png'))), real_images[0])
        
        print(f"\nEditing a real image: {os.path.basename(real_image_path)}")
        apply_blur(real_image_path, os.path.join(OUTPUT_DIR, "real_blurred.png"))
        apply_color_balance(real_image_path, os.path.join(OUTPUT_DIR, "real_color_balanced.png"))
        overlay_text(real_image_path, os.path.join(OUTPUT_DIR, "real_text_overlay.png"))
        add_noise(real_image_path, os.path.join(OUTPUT_DIR, "real_noise.png"))
        apply_compression(real_image_path, os.path.join(OUTPUT_DIR, "real_compressed.jpg"), quality=30)
        apply_rescale(real_image_path, os.path.join(OUTPUT_DIR, "real_rescaled.png"), scale_factor=0.3)
        apply_color_jitter(real_image_path, os.path.join(OUTPUT_DIR, "real_color_jitter.png"))

    # Get a generated AI image to edit
    generated_ai_images = [os.path.join(INPUT_GENERATED_AI_DIR, f) for f in os.listdir(INPUT_GENERATED_AI_DIR) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not generated_ai_images:
        print(f"No generated AI images found in {INPUT_GENERATED_AI_DIR} to edit. Skipping generated AI image edits.")
    else:
        generated_ai_image_path = generated_ai_images[0] # Take the first one
        
        print(f"\nEditing a generated AI image: {os.path.basename(generated_ai_image_path)}")
        apply_blur(generated_ai_image_path, os.path.join(OUTPUT_DIR, "ai_blurred.png"))
        apply_color_balance(generated_ai_image_path, os.path.join(OUTPUT_DIR, "ai_color_balanced.png"))
        overlay_text(generated_ai_image_path, os.path.join(OUTPUT_DIR, "ai_text_overlay.png"))
        add_noise(generated_ai_image_path, os.path.join(OUTPUT_DIR, "ai_noise.png"))
        apply_compression(generated_ai_image_path, os.path.join(OUTPUT_DIR, "ai_compressed.jpg"), quality=30)
        apply_rescale(generated_ai_image_path, os.path.join(OUTPUT_DIR, "ai_rescaled.png"), scale_factor=0.3)
        apply_color_jitter(generated_ai_image_path, os.path.join(OUTPUT_DIR, "ai_color_jitter.png"))

    print("\n--- Image Editing Finished ---")

if __name__ == "__main__":
    main()
