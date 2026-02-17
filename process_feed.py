"""
MerchantPro Product Feed Processor – SAFE VERSION
--------------------------------------------------
• Detectează imagini 1:1 -> skip
• Detectează fundal alb -> padding simplu
• Pentru fundal non-alb -> outpainting cu MASK HARD LOCK
• Produsul NU este modificat niciodată
"""

import os
import io
import base64
import logging
import requests
import numpy as np
import openpyxl
from PIL import Image, ImageDraw
from openai import OpenAI

# ============================================================
# CONFIG
# ============================================================

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
FEED_URL = "https://www.engromag.ro/feed/products/ec219e068cd4552bf8759292992425a6"

IMAGE_SIZE = 1024
AI_TARGET = 5
WHITE_THRESHOLD = 240
SQUARE_TOLERANCE = 0.03

OUTPUT_DIR = "output_safe"
OUTPUT_EXCEL = "rezultate_safe.xlsx"

# ============================================================
# INIT
# ============================================================

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger()
client = OpenAI(api_key=OPENAI_API_KEY)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# UTILITIES
# ============================================================

def download_image(url):
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception as e:
        logger.error(f"Download error: {e}")
        return None


def is_square(img):
    w, h = img.size
    ratio = min(w, h) / max(w, h)
    return ratio >= (1.0 - SQUARE_TOLERANCE)


def detect_border_color(img):
    arr = np.array(img)
    h, w = arr.shape[:2]
    d = 10

    border = np.concatenate([
        arr[:d, :, :].reshape(-1, 3),
        arr[h-d:, :, :].reshape(-1, 3),
        arr[:, :d, :].reshape(-1, 3),
        arr[:, w-d:, :].reshape(-1, 3),
    ])

    avg = border.mean(axis=0)
    rgb = tuple(int(x) for x in avg)
    is_white = all(c > WHITE_THRESHOLD for c in avg)

    return rgb, is_white


def pad_to_square(img, bg_color):
    w, h = img.size
    max_dim = max(w, h)
    canvas = Image.new("RGB", (max_dim, max_dim), bg_color)
    canvas.paste(img, ((max_dim - w)//2, (max_dim - h)//2))
    return canvas.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)


def create_square_with_transparency(img):
    w, h = img.size
    max_dim = max(w, h)

    canvas = Image.new("RGBA", (max_dim, max_dim), (0,0,0,0))
    canvas.paste(img.convert("RGBA"), ((max_dim - w)//2, (max_dim - h)//2))
    return canvas.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)


def create_mask(original_img):
    """
    Alb = editabil
    Negru = protejat
    """
    w, h = original_img.size
    max_dim = max(w, h)

    mask = Image.new("L", (max_dim, max_dim), 255)

    x_offset = (max_dim - w)//2
    y_offset = (max_dim - h)//2

    draw = ImageDraw.Draw(mask)
    draw.rectangle(
        [x_offset, y_offset, x_offset + w, y_offset + h],
        fill=0
    )

    return mask.resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)


def get_bg_description(img):
    arr = np.array(img)
    h, w = arr.shape[:2]
    d = 15

    edges = np.concatenate([
        arr[:d,:,:].reshape(-1,3),
        arr[h-d:,:,:].reshape(-1,3),
        arr[:,:d,:].reshape(-1,3),
        arr[:,w-d:,:].reshape(-1,3),
    ])

    avg = edges.mean(axis=0)
    brightness = avg.mean()

    if brightness > 230:
        return "clean white surface"
    elif brightness > 180:
        return "light neutral surface"
    else:
        return "plain flat surface"


def safe_outpaint(img):

    square = create_square_with_transparency(img)
    mask = create_mask(img)

    img_buf = io.BytesIO()
    square.save(img_buf, format="PNG")
    img_buf.seek(0)
    img_buf.name = "image.png"

    mask_buf = io.BytesIO()
    mask.save(mask_buf, format="PNG")
    mask_buf.seek(0)
    mask_buf.name = "mask.png"

    bg_desc = get_bg_description(img)

    prompt = (
        f"Extend the existing {bg_desc}. "
        f"Only fill the white masked areas. "
        f"The product (black mask) must remain pixel-identical. "
        f"No blending, no color change, no lighting change, "
        f"no object modification."
    )

    try:
        result = client.images.edit(
            model="dall-e-2",
            image=img_buf,
            mask=mask_buf,
            prompt=prompt,
            size=f"{IMAGE_SIZE}x{IMAGE_SIZE}",
            response_format="b64_json",
            n=1,
        )

        data = result.data[0].b64_json
        return Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")

    except Exception as e:
        logger.error(f"AI error: {e}")
        return None


def save_image(img, name):
    safe_name = "".join(c if c.isalnum() or c in ".-_" else "_" for c in name)
    path = os.path.join(OUTPUT_DIR, safe_name)
    img.save(path, format="PNG")
    return os.path.abspath(path)

# ============================================================
# MAIN
# ============================================================

def main():

    logger.info("Downloading feed...")
    resp = requests.get(FEED_URL)
    resp.raise_for_status()

    wb = openpyxl.load_workbook(io.BytesIO(resp.content))
    ws = wb.active

    ws.cell(row=1, column=ws.max_column+1, value="New Image")
    ws.cell(row=1, column=ws.max_column+2, value="Status")

    ai_count = 0

    for row in range(2, ws.max_row+1):

        if ai_count >= AI_TARGET:
            break

        img_url = ws.cell(row=row, column=5).value
        if not img_url:
            continue

        img = download_image(img_url)
        if img is None:
            continue

        if is_square(img):
            logger.info("Skip 1:1")
            continue

        border_color, is_white = detect_border_color(img)

        if is_white:
            logger.info("Padding (white background)")
            result = pad_to_square(img, border_color)
        else:
            logger.info("AI safe outpaint")
            result = safe_outpaint(img)
            if result is None:
                result = pad_to_square(img, border_color)
            else:
                ai_count += 1

        filename = f"product_{row}.png"
        path = save_image(result, filename)

        ws.cell(row=row, column=ws.max_column-1, value=path)
        ws.cell(row=row, column=ws.max_column, value="done")

    output_path = os.path.join(OUTPUT_DIR, OUTPUT_EXCEL)
    wb.save(output_path)

    logger.info(f"Done. Saved to {output_path}")


if __name__ == "__main__":
    main()
