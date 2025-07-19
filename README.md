# Visual-AId-stylegan: Facial Editing with StyleGAN2

A mini facial editing pipeline using StyleGAN2 and latent space manipulation to simulate cosmetic changes (e.g., nose size, jawline, eye openness).

---

## Environment Setup

1. **Python Version**: Recommended Python 3.8 or 3.9 (GPU support best with CUDA 11.x)
2. **(Optional) Create Virtual Environment**:
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
   - For e4e encoder, you may also use `encoder4editing/environment/e4e_env.yaml` with conda if needed.
4. **GPU Support**: For best performance, ensure you have a compatible NVIDIA GPU, CUDA drivers, and the correct PyTorch+CUDA version. See [PyTorch Get Started](https://pytorch.org/get-started/locally/).
5. **Download Pretrained Models**:
   - StyleGAN2 weights: Place in `models/stylegan2/rosinality/checkpoint/` (e.g., `stylegan2-ffhq-config-f.pt`)
   - e4e encoder weights: Place in `models/e4e_ffhq_encode.pt`
   - Latent directions: Place in `models/latent_directions/` (e.g., `smile.npy`, `jawline.npy`)

---

## Project Structure

```
Visual-AId-stylegan/
├── encoder4editing/           # e4e encoder and configs
├── models/
│   ├── stylegan2/rosinality/  # StyleGAN2 implementation and checkpoints
│   └── latent_directions/     # Attribute direction vectors (e.g., smile.npy)
├── scripts/                   # Pipeline scripts
│   ├── generate_samples.py
│   ├── encode_face.py
│   ├── edit_latents.py
│   ├── compare_images.py
│   └── test_pipeline.py
├── notebooks/                 # Jupyter notebooks for exploration
│   └── 01_stylegan_generate.ipynb
├── outputs/                   # Results: latents, reconstructions, edits, etc.
│   ├── latents/
│   ├── reconstructions/
│   ├── edits/
│   └── before_after/
├── requirements.txt
├── README.md
└── project_checklist.txt
```

---

## Usage

### 1. Generate Sample Images

```bash
python scripts/generate_samples.py --num 5 --out outputs/samples
```
- Output: `outputs/samples/sample_1.png`, etc.

### 2. Encode Real Face to Latent Space

```bash
python scripts/encode_face.py --img path/to/your_face.jpg --encoder e4e --out outputs
```
- Output: Latent code in `outputs/latents/`, reconstruction in `outputs/reconstructions/`
- Requires e4e weights (`models/e4e_ffhq_encode.pt`)

### 3. Latent Manipulation (Attribute Editing)

```bash
python scripts/edit_latents.py --latent outputs/latents/your_latent.npy \
    --directions models/latent_directions/smile.npy models/latent_directions/jawline.npy \
    --intensities 2.0 -1.5 --out outputs
```
- Output: Edited latent in `outputs/latents/`, edited image in `outputs/edits/`
- Requires direction files (e.g., `smile.npy`)

### 4. Attribute Editing & Comparison

```bash
python scripts/compare_images.py --original path/to/original.png --edited path/to/edited.png --out outputs/before_after
```
- Output: Side-by-side comparison in `outputs/before_after/`

### 5. Test the Full Pipeline

```bash
python scripts/test_pipeline.py
```
- Runs the above steps in sequence for a test image (requires all weights and direction files).

---

## Current Status

- **StyleGAN2-based pipeline is fully implemented and tested.**
- **Attribute editing, encoding, and before/after visualization are supported.**
- **Project is designed for extensibility to support additional generative models and fine-tuning methods.**

---

## Planned Features

- **Unified webapp UI** for user uploads, edits, and model selection (StyleGAN2, Diffusion, LoRA)
- **Stable Diffusion integration** for advanced text/image-based editing
- **LoRA support** for efficient fine-tuning and custom attribute learning (for both StyleGAN2 and Diffusion)
- **Advanced evaluation and fairness tools** (custom loss functions, metrics, etc.)

---

## References

- [StyleGAN2 (Rosinality)](https://github.com/rosinality/stylegan2-pytorch)
- [e4e: Encoder for Editing](https://github.com/omertov/encoder4editing)
- [InterFaceGAN](https://github.com/shenboli/InterFaceGAN)
- [GANSpace](https://github.com/harskish/ganspace)

---

## Progress Checklist

See `project_checklist.txt` for current progress and next steps.

---

## Troubleshooting

- If you encounter missing module errors, ensure all dependencies are installed and PYTHONPATH includes `encoder4editing/` and `models/stylegan2/rosinality/`.
- For CUDA errors, try running with `--device cpu` or ensure your CUDA drivers and PyTorch versions are compatible.

--- 