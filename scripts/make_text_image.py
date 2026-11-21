"""Generate a 'Text' image like the paper's Text dataset (Appendix E.1):
random strings with random sizes and colors on a white background, 512x512.

python scripts/make_text_image.py --out data/images/text.png --seed 0
"""
import argparse
import random
import string

from PIL import Image, ImageDraw, ImageFont


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/images/text.png")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--lines", type=int, default=12)
    a = p.parse_args()

    rng = random.Random(a.seed)
    img = Image.new("RGB", (a.size, a.size), "white")
    draw = ImageDraw.Draw(img)
    for _ in range(a.lines):
        font_size = rng.randint(14, 48)
        try:
            font = ImageFont.load_default(size=font_size)  # Pillow >= 10.1
        except TypeError:
            font = ImageFont.load_default()
        text = "".join(rng.choice(string.ascii_letters + string.digits) for _ in range(rng.randint(4, 14)))
        color = tuple(rng.randint(0, 200) for _ in range(3))
        xy = (rng.randint(0, a.size - 60), rng.randint(0, a.size - 30))
        draw.text(xy, text, fill=color, font=font)
    img.save(a.out)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
