import sys
sys.path.insert(0, "./src")
print("[Python] Starting test...")

print("[Python] Importing numpy...")
import numpy as np
print("[Python] Numpy imported successfully.")

print("[Python] Importing PyBind11 fast_video_processor...")
import fast_video_processor
print("[Python] PyBind11 module imported successfully.")

print("[Python] Creating dummy arrays...")
prev = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
curr = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
print("[Python] Dummy arrays created.")

print("[Python] Calling compute_physics_flow_cpp...")
try:
    res = fast_video_processor.compute_physics_flow_cpp(prev, curr, 64)
    print(f"[Python] Execution finished. Result shape: {res.shape}")
except Exception as e:
    print(f"[Python] Exception caught: {e}")

print("[Python] Test complete.")
