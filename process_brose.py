"""
Procesare dedicată pentru broșele importante.
Folosește gpt-image-1 Images API cu prompt optimizat pentru extensie naturală.
Trimite imaginea pe canvas transparent - AI completează natural fără decorațiuni.
"""

import os, io, base64, logging, requests, numpy as np, openpyxl
from PIL import Image
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


def extend_to_square_transparent(img):
    """Pune imaginea pe canvas pătrat transparent. Zonele transparente = AI completează."""
    w, h = img.size
    max_dim = max(w, h)
    square = Image.new("RGBA", (max_dim, max_dim), (0, 0, 0, 0))
    img_rgba = img.convert("RGBA")
    square.paste(img_rgba, ((max_dim - w) // 2, (max_dim - h) // 2))
    return square.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)


def get_bg_description(img):
    """Analyze background to create accurate description for prompt."""
    arr = np.array(img)
    h, w = arr.shape[:2]
    d = 15
    # Sample edges
    edges = np.concatenate([
        arr[:d, :, :].reshape(-1, 3),
        arr[h-d:, :, :].reshape(-1, 3),
        arr[d:h-d, :d, :].reshape(-1, 3),
        arr[d:h-d, w-d:, :].reshape(-1, 3),
    ])
    avg = edges.mean(axis=0)
    r, g, b = int(avg[0]), int(avg[1]), int(avg[2])
    brightness = (r + g + b) / 3
    
    if brightness > 230:
        return "white/off-white clean surface"
    elif brightness > 180:
        return "light gray/beige clean surface"
    elif r > g and r > b:
        return "warm-toned surface"
    else:
        return f"plain surface (approximately RGB {r},{g},{b})"


def outpaint_image(img, title):
    """
    gpt-image-1.5 (BEST model) cu quality high — extensie naturală fără deformare.
    Cheia: descriem exact cum arată background-ul și cerem DOAR continuare simplă.
    """
    # Create transparent canvas
    square = extend_to_square_transparent(img)
    
    img_buf = io.BytesIO()
    square.save(img_buf, format="PNG")
    img_buf.seek(0)
    img_buf.name = "image.png"

    bg_desc = get_bg_description(img)
    w, h = img.size

    prompt = (
        f"Fill the transparent areas with a plain, clean {bg_desc}. "
        f"CRITICAL INSTRUCTIONS: "
        f"The transparent areas should contain ONLY the same plain background surface — "
        f"absolutely NO new objects, NO flowers, NO decorative elements, NO patterns, NO embellishments. "
        f"Just extend the flat, clean background surface color and texture. "
        f"Think of it as simply extending the table/surface the product sits on. "
        f"Keep every existing element in the image completely untouched."
    )

    try:
        logger.info(f"  🎨 gpt-image-1.5 HIGH outpainting (bg: {bg_desc})...")
        result = client.images.edit(
            model="gpt-image-1.5",
            image=img_buf,
            prompt=prompt,
            size=f"{IMAGE_SIZE}x{IMAGE_SIZE}",
            quality="high",
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
        logger.error(f"  Eroare API: {e}")
        return None


def main():
    logger.info("=" * 60)
    logger.info("PROCESARE BROSE — gpt-image-1.5 HIGH QUALITY")
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

        # gpt-image-1 outpainting cu prompt optimizat — extensie naturală
        result_img = outpaint_image(img, p["title"])

        if result_img is not None:
            result_img.save(filepath, format="PNG")
            logger.info(f"  ✅ SALVAT: {filepath}")
            results.append({
                "sku": p["sku"],
                "title": p["title"],
                "original": orig_path,
                "ai_result": filepath,
                "status": "success"
            })
        else:
            # Fallback: simple padding with border color
            logger.info(f"  ⚠️ Fallback: padding simplu")
            import numpy as np
            max_dim = max(w, h)
            arr = np.array(img)
            hh, ww = arr.shape[:2]
            d = 10
            border = np.concatenate([arr[:d,:,:].reshape(-1,3), arr[hh-d:,:,:].reshape(-1,3)])
            bg = tuple(int(x) for x in border.mean(axis=0))
            square = Image.new("RGB", (max_dim, max_dim), bg)
            square.paste(img, ((max_dim - w) // 2, (max_dim - h) // 2))
            square = square.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
            square.save(filepath, format="PNG")
            logger.info(f"  💾 Salvat cu padding: {filepath}")
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
