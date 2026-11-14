from pathlib import Path
from PIL import Image, ImageDraw

root = Path("results/figures/shading_transition")
paths = [root / f"r03_b{b:02d}_n14.png" for b in (8, 9, 10)]
images = [Image.open(p).convert("RGB") for p in paths]
width = max(im.width for im in images)
height = max(im.height for im in images)
canvas = Image.new("RGB", (width * 3, height), "white")
for i, im in enumerate(images):
    canvas.paste(im, (i * width, 0))
canvas.save(root / "shading_transition_b08_b09_b10.png")
print(root / "shading_transition_b08_b09_b10.png")
