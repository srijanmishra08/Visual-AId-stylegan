# Facial Editing with StyleGAN2

## Environment Setup

1. **Python Version**: Recommended Python 3.8 or 3.9 (GPU support best with CUDA 11.x)
2. **Create Virtual Environment** (optional but recommended):
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Mac/Linux:
   source venv/bin/activate
   ```
3. **Install Dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. **GPU Support**: For best performance, ensure you have a compatible NVIDIA GPU, CUDA drivers, and the correct PyTorch+CUDA version. See https://pytorch.org/get-started/locally/ for details.
5. **Download Pretrained Models**: See instructions in `notebooks/01_stylegan_generate.ipynb` or the [StyleGAN2 repo](https://github.com/rosinality/stylegan2-pytorch) for model weights.

A mini facial editing pipeline using StyleGAN2 and latent space manipulation to simulate cosmetic changes (e.g., nose size, jawline, eye openness).

## Features
- Upload and encode real face images into StyleGAN2 latent space
- Manipulate facial attributes via latent directions
- Visualize before vs. after edits
- (Optional) Fine-tune with LoRA for diffusion models

## Project Structure
```
facial-editing-gan/
├── data/                  # Datasets (FFHQ, sample faces)
│   ├── ffhq/
│   └── sample_faces/
├── models/                # Pretrained models, latent directions
│   ├── stylegan2/
│   └── latent_directions/
├── notebooks/             # Jupyter notebooks for exploration
│   ├── 01_stylegan_generate.ipynb
│   ├── 02_latent_encode.ipynb
│   ├── 03_attribute_edit.ipynb
├── scripts/               # Python scripts for pipeline/app
│   ├── edit_latents.py
│   └── streamlit_app.py
├── requirements.txt       # Dependencies
├── README.md              # Project overview and instructions
└── outputs/               # Results, logs, visualizations
    ├── before_after/
    └── logs/
```

## Setup
1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Download pretrained models (see notebooks/01_stylegan_generate.ipynb) 

## Generate Sample Images

To generate random face samples using the pretrained StyleGAN2 model:

```bash
python scripts/generate_samples.py --num 5 --out outputs/samples
```

- The script will auto-download FFHQ weights if using `stylegan2_pytorch`.
- Output images will be saved in `outputs/samples/`.
- For Rosinality's implementation, see the placeholder in the script and follow the repo's instructions for model setup. 

## Encode Real Face to Latent Space

To project a real face image into the StyleGAN2 latent space (using e4e or pSp):

```bash
python scripts/encode_face.py --img path/to/your_face.jpg --encoder e4e --out outputs
```

- This will save the latent code as `.npy` in `outputs/latents/` and the reconstructed image in `outputs/reconstructions/`.
- You must have a compatible encoder (e4e or pSp) installed and available to the script. Update the script with the correct import and model path as needed.
- For encoder setup, see the official [e4e repo](https://github.com/omertov/encoder4editing) or [pSp repo](https://github.com/eladrich/pixel2style2pixel). 

## Latent Manipulation (Attribute Editing)

To edit a latent code using one or more attribute directions (e.g., smile, jawline):

```bash
python scripts/edit_latents.py --latent outputs/latents/your_latent.npy \
    --directions models/latent_directions/smile.npy models/latent_directions/jawline.npy \
    --intensities 2.0 -1.5 --out outputs
```

- This will save the manipulated latent as `.npy` in `outputs/latents/` and the edited image in `outputs/edits/`.
- You can specify multiple directions and intensities (must match in number).
- Latent directions can be downloaded from [InterfaceGAN](https://github.com/shenboli/InterFaceGAN) or [GANSpace](https://github.com/harskish/ganspace). 

## Attribute Editing & Comparison

To create a side-by-side comparison of the original and edited images:

```bash
python scripts/compare_images.py --original path/to/original.png --edited path/to/edited.png --out outputs/before_after
```

- The script will save a new image showing both versions side-by-side in `outputs/before_after/`. 