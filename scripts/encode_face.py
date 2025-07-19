import os
import argparse
import numpy as np
from PIL import Image
import torch
import sys

# Add e4e and Rosinality paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../encoder4editing')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/stylegan2/rosinality')))

from utils.model_utils import setup_model
from utils.common import tensor2im
from configs import transforms_config
from models.stylegan2.model import Generator


def encode_image(img_path, e4e_ckpt, stylegan_ckpt, out_dir, device='cuda'):
    os.makedirs(os.path.join(out_dir, 'latents'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'reconstructions'), exist_ok=True)
    img_name = os.path.splitext(os.path.basename(img_path))[0]

    # Load e4e model
    net, opts = setup_model(e4e_ckpt, device)
    net.eval()

    # Preprocess image as e4e expects
    transforms = transforms_config.EncodeTransforms(opts).get_transforms()
    transform = transforms['transform_inference']
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)

    # Encode image
    with torch.no_grad():
        latent = net.encoder(img_tensor)
        if opts.start_from_latent_avg:
            latent = latent + net.latent_avg.repeat(latent.shape[0], 1, 1)
        # Save latent as numpy
        latent_np = latent.squeeze(0).cpu().numpy()
        latent_path = os.path.join(out_dir, 'latents', f'{img_name}_latent.npy')
        np.save(latent_path, latent_np)

        # Reconstruct image using StyleGAN2 generator
        # Load generator
        g_ema = Generator(opts.stylegan_size, 512, 8, channel_multiplier=2).to(device)
        ckpt = torch.load(stylegan_ckpt, map_location=device)
        g_ema.load_state_dict(ckpt['g_ema'], strict=False)
        g_ema.eval()
        # Generate image from latent
        with torch.no_grad():
            img_gen, _ = g_ema([latent], input_is_latent=True, randomize_noise=False, return_latents=True)
            img_gen = net.face_pool(img_gen)
            recon_img = tensor2im(img_gen[0].cpu())
            recon_path = os.path.join(out_dir, 'reconstructions', f'{img_name}_recon.png')
            recon_img.save(recon_path)

    print(f"Saved latent to {latent_path} and reconstruction to {recon_path}")

def main():
    parser = argparse.ArgumentParser(description="Encode a real face image into StyleGAN2 latent space using e4e.")
    parser.add_argument("--img", type=str, required=True, help="Path to input image.")
    parser.add_argument("--out", type=str, default="outputs", help="Output directory.")
    parser.add_argument("--e4e_ckpt", type=str, default="models/e4e_ffhq_encode.pt", help="Path to e4e checkpoint.")
    parser.add_argument("--stylegan_ckpt", type=str, default="models/stylegan2/rosinality/checkpoint/stylegan2-ffhq-config-f.pt", help="Path to StyleGAN2 generator checkpoint.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu).")
    args = parser.parse_args()
    encode_image(args.img, args.e4e_ckpt, args.stylegan_ckpt, args.out, args.device)

if __name__ == "__main__":
    main() 