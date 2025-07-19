import os
import argparse
from PIL import Image

try:
    from stylegan2_pytorch import ModelLoader
    STYLEGAN2_AVAILABLE = True
except ImportError:
    STYLEGAN2_AVAILABLE = False
    print("stylegan2_pytorch not installed. Please install or use Rosinality's implementation.")


def generate_samples(num_samples=5, out_dir="outputs/samples", seed=42):
    os.makedirs(out_dir, exist_ok=True)
    if STYLEGAN2_AVAILABLE:
        loader = ModelLoader(
            base_dir="models/stylegan2",
            name="ffhq-1024",
            download=True  # auto-download weights
        )
        G = loader.G
        for i in range(num_samples):
            img = loader.generate_images(1, seed=seed+i)[0]
            img.save(os.path.join(out_dir, f"sample_{i+1}.png"))
        print(f"Saved {num_samples} samples to {out_dir}")
    else:
        print("[Placeholder] Please implement sample generation using Rosinality's StyleGAN2 if needed.")


def main():
    parser = argparse.ArgumentParser(description="Generate random face samples using StyleGAN2.")
    parser.add_argument("--num", type=int, default=5, help="Number of samples to generate.")
    parser.add_argument("--out", type=str, default="outputs/samples", help="Output directory.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()
    generate_samples(args.num, args.out, args.seed)

if __name__ == "__main__":
    main() 