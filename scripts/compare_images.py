import os
import argparse
from PIL import Image

def compare_images(original_path, edited_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    orig = Image.open(original_path).convert('RGB')
    edit = Image.open(edited_path).convert('RGB')
    # Resize to same height
    if orig.size[1] != edit.size[1]:
        edit = edit.resize((int(edit.size[0] * orig.size[1] / edit.size[1]), orig.size[1]))
    # Concatenate horizontally
    total_width = orig.size[0] + edit.size[0]
    max_height = max(orig.size[1], edit.size[1])
    comp = Image.new('RGB', (total_width, max_height))
    comp.paste(orig, (0, 0))
    comp.paste(edit, (orig.size[0], 0))
    # Save
    base_name = os.path.splitext(os.path.basename(original_path))[0]
    out_path = os.path.join(out_dir, f'{base_name}_vs_edited.png')
    comp.save(out_path)
    print(f'Saved comparison image to {out_path}')

def main():
    parser = argparse.ArgumentParser(description='Compare original and edited images side-by-side.')
    parser.add_argument('--original', type=str, required=True, help='Path to original image.')
    parser.add_argument('--edited', type=str, required=True, help='Path to edited image.')
    parser.add_argument('--out', type=str, default='outputs/before_after', help='Output directory.')
    args = parser.parse_args()
    compare_images(args.original, args.edited, args.out)

if __name__ == '__main__':
    main() 