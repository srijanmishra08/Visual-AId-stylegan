import os
import argparse
import numpy as np
from PIL import Image

try:
    from stylegan2_pytorch import ModelLoader
    STYLEGAN2_AVAILABLE = True
except ImportError:
    STYLEGAN2_AVAILABLE = False

def edit_latent(latent_path, direction_paths, intensities, out_dir):
    os.makedirs(os.path.join(out_dir, 'latents'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'edits'), exist_ok=True)
    latent = np.load(latent_path)
    latent_name = os.path.splitext(os.path.basename(latent_path))[0]

    # Load and apply directions
    edited_latent = latent.copy()
    for dir_path, alpha in zip(direction_paths, intensities):
        direction = np.load(dir_path)
        edited_latent = edited_latent + alpha * direction

    # Save new latent
    edited_latent_path = os.path.join(out_dir, 'latents', f'{latent_name}_edited.npy')
    np.save(edited_latent_path, edited_latent)

    # Generate and save edited image
    if STYLEGAN2_AVAILABLE:
        loader = ModelLoader(base_dir="models/stylegan2", name="ffhq-1024", download=True)
        G = loader.G
        edited_img = loader.generate_from_latent(edited_latent)
        edited_img.save(os.path.join(out_dir, 'edits', f'{latent_name}_edited.png'))
        print(f"Saved edited image to {os.path.join(out_dir, 'edits', f'{latent_name}_edited.png')}")
    else:
        print("StyleGAN2 not available. Please install stylegan2_pytorch or use Rosinality's implementation.")

    print(f"Saved edited latent to {edited_latent_path}")

def main():
    parser = argparse.ArgumentParser(description="Edit a latent code using one or more directions.")
    parser.add_argument("--latent", type=str, required=True, help="Path to input latent .npy file.")
    parser.add_argument("--directions", type=str, nargs='+', required=True, help="Path(s) to direction .npy file(s).")
    parser.add_argument("--intensities", type=float, nargs='+', required=True, help="Intensity for each direction.")
    parser.add_argument("--out", type=str, default="outputs", help="Output directory.")
    args = parser.parse_args()
    if len(args.directions) != len(args.intensities):
        raise ValueError("Number of directions and intensities must match.")
    edit_latent(args.latent, args.directions, args.intensities, args.out)

if __name__ == "__main__":
    main() 