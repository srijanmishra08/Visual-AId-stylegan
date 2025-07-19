import os
import subprocess
import numpy as np

def run_cmd(cmd):
    print(f"\n[TEST] Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"[ERROR] {result.stderr}")
    return result.returncode

def test_sample_generation():
    print("\n=== Test: Sample Generation ===")
    out_dir = "outputs/test_samples"
    rc = run_cmd(f"python scripts/generate_samples.py --num 2 --out {out_dir}")
    for i in range(1, 3):
        img_path = os.path.join(out_dir, f"sample_{i}.png")
        print(f"Checking {img_path}: {'OK' if os.path.exists(img_path) else 'MISSING'}")

def test_encoding_edge_cases():
    print("\n=== Test: Encoding Edge Cases ===")
    # Missing image
    rc = run_cmd("python scripts/encode_face.py --img not_a_real_file.jpg --encoder e4e --out outputs/test_encode")
    # Invalid encoder
    rc = run_cmd("python scripts/encode_face.py --img outputs/test_samples/sample_1.png --encoder notarealencoder --out outputs/test_encode")

def test_latent_editing_edge_cases():
    print("\n=== Test: Latent Editing Edge Cases ===")
    # Missing latent
    rc = run_cmd("python scripts/edit_latents.py --latent not_a_real_latent.npy --directions models/latent_directions/smile.npy --intensities 2.0 --out outputs/test_edit")
    # Mismatched directions/intensities
    rc = run_cmd("python scripts/edit_latents.py --latent outputs/latents/your_latent.npy --directions models/latent_directions/smile.npy models/latent_directions/jawline.npy --intensities 2.0 --out outputs/test_edit")
    # Missing direction
    rc = run_cmd("python scripts/edit_latents.py --latent outputs/latents/your_latent.npy --directions not_a_real_direction.npy --intensities 1.0 --out outputs/test_edit")

def test_pipeline_happy_path():
    print("\n=== Test: Pipeline Happy Path ===")
    # Generate sample
    out_dir = "outputs/test_pipeline"
    run_cmd(f"python scripts/generate_samples.py --num 1 --out {out_dir}")
    sample_img = os.path.join(out_dir, "sample_1.png")
    # Encode sample (if encoder available)
    if os.path.exists(sample_img):
        run_cmd(f"python scripts/encode_face.py --img {sample_img} --encoder e4e --out {out_dir}")
        latent_path = os.path.join(out_dir, "latents", "sample_1_latent.npy")
        # Edit latent (if latent exists and direction exists)
        if os.path.exists(latent_path):
            direction_path = "models/latent_directions/smile.npy"
            if os.path.exists(direction_path):
                run_cmd(f"python scripts/edit_latents.py --latent {latent_path} --directions {direction_path} --intensities 2.0 --out {out_dir}")
            else:
                print(f"[SKIP] Direction file {direction_path} not found.")
        else:
            print(f"[SKIP] Latent file {latent_path} not found.")
    else:
        print(f"[SKIP] Sample image {sample_img} not found.")

def main():
    test_sample_generation()
    test_encoding_edge_cases()
    test_latent_editing_edge_cases()
    test_pipeline_happy_path()

if __name__ == "__main__":
    main() 