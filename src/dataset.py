import os
import argparse
from PIL import Image


def parse_args():
  parser = argparse.ArgumentParser(description="Preprocess satellite/drone images for LoRA training.")
  parser.add_argument(
    "--input_dir", type=str, required=True,
    help="Directory containing raw images"
  )
  parser.add_argument(
    "--output_dir", type=str, required=True,
    help="Directory to save processed images"
  )
  parser.add_argument(
    "--resolution", type=int, default=512,
    help="Target image width and height (e.g., 512)"
  )
  parser.add_argument(
    "--prompts_csv", type=str, default=None,
    help="Optional CSV file mapping filenames to text prompts"
  )
  return parser.parse_args()


def ensure_dir(path):
  if not os.path.exists(path):
    os.makedirs(path)


def load_prompts(csv_path):
  prompts = {}
  with open(csv_path, 'r', encoding='utf-8') as f:
    # Expect CSV with header: filename,prompt
    for i, line in enumerate(f):
      if i == 0:
        continue
      parts = line.strip().split(',')
      if len(parts) >= 2:
        prompts[parts[0]] = ",".join(parts[1:])
  return prompts


def process_images(input_dir, output_dir, resolution, prompts=None):
  supported_exts = {'.jpg', '.jpeg', '.png', '.tiff', '.bmp'}
  for fname in os.listdir(input_dir):
    ext = os.path.splitext(fname)[1].lower()
    if ext not in supported_exts:
      continue
    in_path = os.path.join(input_dir, fname)
    out_path = os.path.join(output_dir, fname)

    try:
      img = Image.open(in_path).convert('RGB')
      img = img.resize((resolution, resolution), Image.LANCZOS)
      img.save(out_path, format='PNG')

      if prompts is not None and fname in prompts:
        # Save prompt alongside image if needed
        prompt = prompts[fname]
        # Write to a .txt file next to image
        txt_path = os.path.splitext(out_path)[0] + '.txt'
        with open(txt_path, 'w', encoding='utf-8') as tf:
          tf.write(prompt)

      print(f"Processed {fname} -> {out_path}")
    except Exception as e:
      print(f"Error processing {fname}: {e}")


def main():
  args = parse_args()
  ensure_dir(args.output_dir)

  prompts = None
  if args.prompts_csv:
    prompts = load_prompts(args.prompts_csv)

  process_images(
    input_dir=args.input_dir,
    output_dir=args.output_dir,
    resolution=args.resolution,
    prompts=prompts
  )


if __name__ == '__main__':
  main()
