"""
Procesare dedicată pentru broșele importante.
Descarcă imaginile, le procesează cu AI outpainting, salvează în output/.
"""

import os, io, base64, logging, requests, numpy as np, openpyxl
from PIL import Image, ImageDraw
from openai import OpenAI

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
FEED_URL = "https://www.engromag.ro/feed/products/ec219e068cd4552bf8759292992425a6"
IMAGE_SIZE = 1024
OUTPUT_DIR = "output"

# SKU-urile importante - broșele
TARGET_SKUS = {"9991943", "9991929", "9991930", "9991931", "9991932"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)
client = OpenAI(api_key=OPENAI_API_KEY)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def download_image(url):
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception as e:
        logger.error(f"  Eroare descarcare: {e}")
        return None


def get_border_color(img):
    arr = np.array(img)
    h, w = arr.shape[:2]
    d = 10
    border = np.concatenate([
        arr[:d, :, :].reshape(-1, 3),
        arr[h-d:, :, :].reshape(-1, 3),
        arr[d:h-d, :d, :].reshape(-1, 3),
        arr[d:h-d, w-d:, :].reshape(-1, 3),
    ])
    avg = border.mean(axis=0)
    return (int(round(avg[0])), int(round(avg[1])), int(round(avg[2])))


def extend_to_square_transparent(img):
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
        # Top edge of original
        if oy + i < max_dim:
            arr[oy + i, ox:ox+w, 3] = np.minimum(arr[oy + i, ox:ox+w, 3], alpha)
        # Bottom edge of original
        if oy + h - 1 - i >= 0:
            arr[oy + h - 1 - i, ox:ox+w, 3] = np.minimum(arr[oy + h - 1 - i, ox:ox+w, 3], alpha)
        # Left edge of original
        if ox + i < max_dim:
            arr[oy:oy+h, ox + i, 3] = np.minimum(arr[oy:oy+h, ox + i, 3], alpha)
        # Right edge of original
        if ox + w - 1 - i >= 0:
            arr[oy:oy+h, ox + w - 1 - i, 3] = np.minimum(arr[oy:oy+h, ox + w - 1 - i, 3], alpha)
    square = Image.fromarray(arr, "RGBA")
    return square.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)


def outpaint_image(img_square, title):
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
        return None
    except Exception as e:
        logger.error(f"  Eroare OpenAI: {e}")
        return None


def main():
    logger.info("=" * 60)
    logger.info("PROCESARE BROSE IMPORTANTE")
    logger.info(f"  SKU-uri: {TARGET_SKUS}")
    logger.info("=" * 60)

    # Download feed
    logger.info("Descarc feed-ul...")
    resp = requests.get(FEED_URL, timeout=30)
    resp.raise_for_status()
    wb = openpyxl.load_workbook(io.BytesIO(resp.content))
    ws = wb.active

    # Find target products
    found = []
    for row_idx in range(2, ws.max_row + 1):
        sku = str(ws.cell(row=row_idx, column=2).value or "").strip()
        if sku in TARGET_SKUS:
            title = str(ws.cell(row=row_idx, column=3).value or "")
            img_url = str(ws.cell(row=row_idx, column=5).value or "")
            product_id = ws.cell(row=row_idx, column=1).value
            found.append({"sku": sku, "title": title, "img_url": img_url, "id": product_id, "row": row_idx})
            logger.info(f"  Gasit: SKU {sku} - {title[:60]}")

    logger.info(f"\nGasite {len(found)}/{len(TARGET_SKUS)} produse")

    # Process each
    results = []
    for i, p in enumerate(found, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"[{i}/{len(found)}] SKU: {p['sku']} - {p['title'][:60]}")

        filename = f"brosa_{p['sku']}.png"
        filepath = os.path.join(OUTPUT_DIR, filename)

        # RESUME: skip if already exists
        if os.path.exists(filepath):
            logger.info(f"  ⏭️  SKIP - deja exista: {filepath}")
            results.append({"sku": p["sku"], "title": p["title"], "original": "", "ai_result": filepath, "status": "resumed"})
            continue

        # Download image
        img_url = p["img_url"].split(";")[0].strip() if p["img_url"] else ""
        if not img_url:
            logger.error("  Fara URL imagine!")
            continue

        img = download_image(img_url)
        if img is None:
            continue

        w, h = img.size
        logger.info(f"  Dimensiune: {w}x{h}")

        # Save original for comparison
        orig_filename = f"brosa_{p['sku']}_original.png"
        orig_path = os.path.join(OUTPUT_DIR, orig_filename)
        img.save(orig_path, format="PNG")

        # Extend to square with transparent edges (AI fills transparent areas)
        square_img = extend_to_square_transparent(img)

        # AI outpainting with gpt-image-1
        result_img = outpaint_image(square_img, p["title"])

        if result_img is not None:
            # COMPOSITE: paste original back on top so AI NEVER affects the product
            max_dim = max(w, h)
            # Scale original to match the proportion inside IMAGE_SIZE
            scale = IMAGE_SIZE / max_dim
            orig_resized = img.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
            ox = (IMAGE_SIZE - orig_resized.width) // 2
            oy = (IMAGE_SIZE - orig_resized.height) // 2
            result_img.paste(orig_resized, (ox, oy))
            logger.info(f"  🔒 Original lipit înapoi - produs protejat 100%")

            result_img.save(filepath, format="PNG")
            logger.info(f"  ✅ REUSIT! Salvat: {filepath}")
            results.append({
                "sku": p["sku"],
                "title": p["title"],
                "original": orig_path,
                "ai_result": filepath,
                "status": "success"
            })
        else:
            square_img.save(filepath, format="PNG")
            logger.info(f"  ❌ ESUAT - salvat cu padding alb: {filepath}")
            results.append({
                "sku": p["sku"],
                "title": p["title"],
                "original": orig_path,
                "ai_result": filepath,
                "status": "failed"
            })

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"REZUMAT BROSE")
    logger.info(f"  Procesate: {len(results)}")
    logger.info(f"  Reusit AI: {sum(1 for r in results if r['status']=='success')}")
    logger.info(f"  Esuat:     {sum(1 for r in results if r['status']=='failed')}")
    logger.info(f"{'='*60}")

    return results


if __name__ == "__main__":
    main()
