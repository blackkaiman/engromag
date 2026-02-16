"""
MerchantPro Product Feed Processor (Excel)
- Downloads product images from XLSX feed
- Detects white backgrounds -> skip
- Detects already 1:1 images -> skip
- Uses OpenAI outpainting for non-white, non-square backgrounds
- Saves results in new Excel file with status + new image path
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
# CONFIGURATION
# ============================================================
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
FEED_URL = "https://www.engromag.ro/feed/products/ec219e068cd4552bf8759292992425a6"
AI_TARGET = 50  # Stop after generating this many AI outpainted images
IMAGE_SIZE = 1024
WHITE_THRESHOLD = 240
BORDER_SAMPLE_DEPTH = 10
SQUARE_TOLERANCE = 0.03  # 3% tolerance for 1:1
OUTPUT_DIR = "output"
OUTPUT_EXCEL = "rezultate.xlsx"

# ============================================================
# SETUP
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)
client = OpenAI(api_key=OPENAI_API_KEY)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# IMAGE FUNCTIONS
# ============================================================

def download_image(url: str) -> Image.Image | None:
    """Download an image from URL and return as PIL Image."""
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception as e:
        logger.error(f"  Eroare descarcare: {e}")
        return None


def is_square(img: Image.Image) -> bool:
    """Check if image is already 1:1."""
    w, h = img.size
    ratio = min(w, h) / max(w, h)
    return ratio >= (1.0 - SQUARE_TOLERANCE)


def get_border_color(img: Image.Image) -> tuple:
    """Get average border color as (R, G, B) tuple."""
    arr = np.array(img)
    h, w = arr.shape[:2]
    d = BORDER_SAMPLE_DEPTH

    border = np.concatenate([
        arr[:d, :, :].reshape(-1, 3),
        arr[h-d:, :, :].reshape(-1, 3),
        arr[d:h-d, :d, :].reshape(-1, 3),
        arr[d:h-d, w-d:, :].reshape(-1, 3),
    ])

    avg = border.mean(axis=0)
    color = (int(round(avg[0])), int(round(avg[1])), int(round(avg[2])))
    is_white = all(c > WHITE_THRESHOLD for c in avg)
    logger.info(f"  Border RGB: {color} -> {'ALB' if is_white else 'NU e alb'}")
    return color, is_white


def extend_to_square(img: Image.Image, bg_color: tuple = None) -> Image.Image:
    """Extend image to square with matching background color, resize to IMAGE_SIZE (for padding-only)."""
    w, h = img.size
    if bg_color is None:
        bg_color, _ = get_border_color(img)
    max_dim = max(w, h)
    square = Image.new("RGB", (max_dim, max_dim), bg_color)
    square.paste(img, ((max_dim - w) // 2, (max_dim - h) // 2))
    return square.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)


def extend_to_square_transparent(img: Image.Image) -> Image.Image:
    """Extend image to square with transparent areas where AI should generate."""
    w, h = img.size
    max_dim = max(w, h)
    # RGBA with transparent background
    square = Image.new("RGBA", (max_dim, max_dim), (0, 0, 0, 0))
    img_rgba = img.convert("RGBA")
    square.paste(img_rgba, ((max_dim - w) // 2, (max_dim - h) // 2))
    # Add tiny feathered edge for smooth blending (minimal to preserve original content)
    fade = 2  # pixels of feather - keep very small to not eat into original
    arr = np.array(square)
    ox, oy = (max_dim - w) // 2, (max_dim - h) // 2
    for i in range(fade):
        alpha = int(255 * (i / fade))
        if oy + i < max_dim:
            arr[oy + i, ox:ox+w, 3] = np.minimum(arr[oy + i, ox:ox+w, 3], alpha)
        if oy + h - 1 - i >= 0:
            arr[oy + h - 1 - i, ox:ox+w, 3] = np.minimum(arr[oy + h - 1 - i, ox:ox+w, 3], alpha)
        if ox + i < max_dim:
            arr[oy:oy+h, ox + i, 3] = np.minimum(arr[oy:oy+h, ox + i, 3], alpha)
        if ox + w - 1 - i >= 0:
            arr[oy:oy+h, ox + w - 1 - i, 3] = np.minimum(arr[oy:oy+h, ox + w - 1 - i, 3], alpha)
    square = Image.fromarray(arr, "RGBA")
    return square.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)


def outpaint_image(img_square: Image.Image, title: str) -> Image.Image | None:
    """Use OpenAI gpt-image-1 to outpaint. Image has transparent areas = AI fills."""
    img_buf = io.BytesIO()
    img_square.save(img_buf, format="PNG")
    img_buf.seek(0)
    img_buf.name = "image.png"

    prompt = (
        f"Extend the background of this product photo to fill the transparent areas of the square canvas. "
        f"CRITICAL: PRESERVE every single element that already exists in the image - do NOT remove, alter, or simplify "
        f"any existing objects, decorations, flowers, patterns, textures or background details. "
        f"Keep the product and ALL existing background elements exactly as they are. "
        f"Only fill the transparent/empty areas by continuing the same background style, colors, textures and patterns "
        f"that are visible at the edges of the existing image. "
        f"The result should look like the original photo was simply taken with a wider frame."
    )

    try:
        logger.info(f"  Apel OpenAI gpt-image-1 outpainting...")
        result = client.images.edit(
            model="gpt-image-1",
            image=img_buf,
            prompt=prompt,
            size=f"{IMAGE_SIZE}x{IMAGE_SIZE}",
            n=1,
        )

        if result.data:
            d = result.data[0]
            if hasattr(d, 'b64_json') and d.b64_json:
                return Image.open(io.BytesIO(base64.b64decode(d.b64_json))).convert("RGB")
            elif hasattr(d, 'url') and d.url:
                return Image.open(io.BytesIO(requests.get(d.url, timeout=30).content)).convert("RGB")

        logger.error("  Raspuns gol de la API")
        return None
    except Exception as e:
        logger.error(f"  Eroare OpenAI: {e}")
        return None


def save_img(img: Image.Image, name: str) -> str:
    """Save image to output dir, return absolute path."""
    safe = "".join(c if c.isalnum() or c in ".-_" else "_" for c in name)
    path = os.path.join(OUTPUT_DIR, safe)
    img.save(path, format="PNG")
    return os.path.abspath(path)


# ============================================================
# MAIN
# ============================================================

def main():
    logger.info("=" * 60)
    logger.info("MerchantPro Feed Processor")
    logger.info(f"  Target AI: {AI_TARGET} | Size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    logger.info("=" * 60)

    # 1. Download feed
    logger.info(f"Descarc feed-ul...")
    resp = requests.get(FEED_URL, timeout=30)
    resp.raise_for_status()
    wb = openpyxl.load_workbook(io.BytesIO(resp.content))
    ws = wb.active

    logger.info(f"Feed: {ws.max_row - 1} produse, {ws.max_column} coloane")

    # Find columns
    headers = {}
    for col in range(1, ws.max_column + 1):
        h = str(ws.cell(row=1, column=col).value or "").strip()
        headers[h] = col

    col_id = headers.get("ID produs", 1)
    col_sku = headers.get("Cod produs - SKU", 2)
    col_title = headers.get("Nume produs", 3)
    col_desc = headers.get("Descriere produs", 4)
    col_img = headers.get("Imagini", 5)

    # Add output columns
    col_new_img = ws.max_column + 1
    col_status = ws.max_column + 2
    col_reason = ws.max_column + 3

    ws.cell(row=1, column=col_new_img, value="Imagine Noua (path)")
    ws.cell(row=1, column=col_status, value="Status")
    ws.cell(row=1, column=col_reason, value="Motiv")

    # 2. Scan existing output for resume
    existing_files = set(os.listdir(OUTPUT_DIR)) if os.path.exists(OUTPUT_DIR) else set()
    logger.info(f"Fisiere existente in output/: {len(existing_files)}")

    # 2b. Process products
    counts = {"skip": 0, "ai_generated": 0, "ai_failed": 0, "resumed": 0}

    for row_idx in range(2, ws.max_row + 1):
        # Stop when we have enough AI generated images
        if counts["ai_generated"] >= AI_TARGET:
            logger.info(f"\n{'='*60}")
            logger.info(f"TARGET ATINS: {AI_TARGET} imagini AI generate!")
            break

        product_id = ws.cell(row=row_idx, column=col_id).value
        sku = ws.cell(row=row_idx, column=col_sku).value
        title = str(ws.cell(row=row_idx, column=col_title).value or "")
        desc = str(ws.cell(row=row_idx, column=col_desc).value or "")
        img_url = str(ws.cell(row=row_idx, column=col_img).value or "")

        nr = row_idx - 1
        filename = f"product_{nr:03d}_{sku or product_id}.png"

        # RESUME: skip if file already exists in output/
        if filename in existing_files:
            counts["resumed"] += 1
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"[{nr}] (AI: {counts['ai_generated']}/{AI_TARGET}) {title[:70]}")
        logger.info(f"  SKU: {sku} | ID: {product_id}")

        if not img_url:
            logger.warning(f"  Fara imagine!")
            ws.cell(row=row_idx, column=col_status, value="skip")
            ws.cell(row=row_idx, column=col_reason, value="Fara URL imagine")
            counts["skip"] += 1
            continue

        # Download
        img = download_image(img_url)
        if img is None:
            ws.cell(row=row_idx, column=col_status, value="ai_failed")
            ws.cell(row=row_idx, column=col_reason, value="Eroare descarcare imagine")
            counts["ai_failed"] += 1
            continue

        w, h = img.size
        logger.info(f"  Dimensiune: {w}x{h}")

        # Check: Already square? -> just resize, no AI needed
        if is_square(img):
            logger.info(f"  -> SKIP: Imaginea e deja 1:1 ({w}x{h})")
            square = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
            path = save_img(square, filename)
            ws.cell(row=row_idx, column=col_new_img, value=path)
            ws.cell(row=row_idx, column=col_status, value="skip")
            ws.cell(row=row_idx, column=col_reason, value=f"Imaginea e deja 1:1 ({w}x{h})")
            counts["skip"] += 1
            continue

        # Non-square: check background color
        border_color, is_white = get_border_color(img)

        # Non-square + white/near-white bg -> pad with matching border color (no AI)
        if is_white:
            logger.info(f"  -> PADDING cu culoarea bordurii {border_color} ({w}x{h})")
            square_img = extend_to_square(img, bg_color=border_color)
            path = save_img(square_img, filename)
            ws.cell(row=row_idx, column=col_new_img, value=path)
            ws.cell(row=row_idx, column=col_status, value="padded")
            ws.cell(row=row_idx, column=col_reason, value=f"Padding cu culoare bordura {border_color} ({w}x{h})")
            counts["skip"] += 1
            continue

        # Non-square + non-white bg -> AI Outpainting
        logger.info(f"  -> AI OUTPAINTING (non-patrat {w}x{h}, fundal colorat)")
        square_img = extend_to_square_transparent(img)

        result_img = outpaint_image(square_img, title)

        if result_img is not None:
            # COMPOSITE: paste original back on top so AI NEVER affects the product
            max_dim = max(w, h)
            scale = IMAGE_SIZE / max_dim
            orig_resized = img.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
            ox = (IMAGE_SIZE - orig_resized.width) // 2
            oy = (IMAGE_SIZE - orig_resized.height) // 2
            result_img.paste(orig_resized, (ox, oy))
            logger.info(f"  🔒 Original lipit înapoi - produs protejat 100%")

            path = save_img(result_img, filename)
            ws.cell(row=row_idx, column=col_new_img, value=path)
            ws.cell(row=row_idx, column=col_status, value="ai_generated")
            ws.cell(row=row_idx, column=col_reason, value=f"Outpainting AI reusit ({w}x{h} -> 1024x1024)")
            counts["ai_generated"] += 1
            logger.info(f"  REUSIT!")
        else:
            path = save_img(square_img, filename)
            ws.cell(row=row_idx, column=col_new_img, value=path)
            ws.cell(row=row_idx, column=col_status, value="ai_failed")
            ws.cell(row=row_idx, column=col_reason, value="Eroare API OpenAI - salvat cu padding alb")
            counts["ai_failed"] += 1
            logger.info(f"  ESUAT - salvat cu padding alb")

    # 3. Save results
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_EXCEL)
    wb.save(output_path)

    # Summary
    total = sum(counts.values())
    logger.info(f"\n{'='*60}")
    logger.info(f"REZUMAT FINAL")
    logger.info(f"  Total procesate: {total}")
    logger.info(f"  Resumed (skip):  {counts['resumed']}")
    logger.info(f"  Skip (alb/1:1):  {counts['skip']}")
    logger.info(f"  AI generate:     {counts['ai_generated']}")
    logger.info(f"  AI failed:       {counts['ai_failed']}")
    logger.info(f"")
    logger.info(f"  Excel salvat: {os.path.abspath(output_path)}")
    logger.info(f"  Imagini in:   {os.path.abspath(OUTPUT_DIR)}/")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
