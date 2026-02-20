import os
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, '..', 'data', 'generated_ai', 'images')
MODEL_ID = "runwayml/stable-diffusion-v1-5" # A popular and widely used Stable Diffusion model
NUM_INFERENCE_STEPS = 50 # Number of denoising steps

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_images(prompts, output_dir, model_id, num_inference_steps=50):
    """
    Generates images using a pre-trained Stable Diffusion model.

    Args:
        prompts (list): A list of text prompts for image generation.
        output_dir (str): Directory to save the generated images.
        model_id (str): Hugging Face model ID for the Stable Diffusion pipeline.
        num_inference_steps (int): Number of denoising steps. Higher generally means better quality but slower.
    """
    print(f"Loading Stable Diffusion pipeline from {model_id}...")
    # Using float16 for potentially faster inference and lower memory usage if GPU is available
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
    
    # Move model to GPU if available
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
        print("Model moved to GPU.")
    else:
        print("CUDA not available, running on CPU. This will be slower.")

    print(f"Generating {len(prompts)} images...")
    for i, prompt in enumerate(prompts):
        print(f"Generating image for prompt: '{prompt}'")
        image = pipe(prompt, num_inference_steps=num_inference_steps, height=256, width=256).images[0]
        
        # Save the image
        output_path = os.path.join(output_dir, f"generated_image_{i+1}.png")
        image.save(output_path)
        print(f"Saved to {output_path}")

def main():
    """Main function to run image generation."""
    
    # Example prompts - these can be expanded or made more dynamic
    prompts = [
        # Face swapping - age/gender/mismatched
        "Photorealistic portrait, young woman with an old man's eyes, slight facial distortion, high detail",
        "Portrait of an elderly man with a youthful woman's facial features, subtly unnatural lighting, low quality swap",
        "Close-up of a person with one side of their face belonging to a different gender, seamless but uncanny, realistic digital art",
        "Two people in a natural setting, their faces subtly swapped with each other, realistic but with minor artifacts",

        # Partial face swaps
        "Portrait of a woman with a man's mouth perfectly grafted, uncanny valley, photorealistic",
        "Close-up of a face with eyes from a different person, subtle color mismatch, high resolution",

        # Body and head swaps / Background swapping
        "Head of a CEO on the body of a ballet dancer, standing in a boardroom, original background retained, slightly unnatural blend",
        "A person standing in front of the Eiffel Tower, but the background is clearly a different scene (e.g., a desert), high fidelity swap",
        "Athlete's body with a scholar's head, high quality photography, subtle skin tone differences",

        # Expression and pose swapping
        "Portrait of a person with a forced, unnatural smile expression, subtle facial glitches, realistic photo",
        "A person sitting, but their body pose seems mismatched with their head orientation and expression, high detail photography",

        # High-quality and low-quality swaps (generic)
        "Ultra-realistic portrait, perfectly executed face swap, almost undetectable, high resolution photo",
        "Portrait with clear signs of low quality face swap, blurry edges, color discrepancies, digital artifacts",
        "Two faces merged into one, uncanny and distorted, low resolution digital image with compression artifacts",
        "Photorealistic image of a person, but with a subtle texture overlay making it look manipulated, high quality"
    ]

    print("--- AI Image Generation Script ---")
    generate_images(prompts, OUTPUT_DIR, MODEL_ID, NUM_INFERENCE_STEPS)
    print("\n--- Image Generation Finished ---")

if __name__ == "__main__":
    main()
