import base64
import io
import re
import os

import requests
from PIL import Image, ImageOps
from flask import Flask, jsonify, request
from openai import OpenAI

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10MB uploads

MODEL = os.environ.get("NVIDIA_MODEL", "deepseek-ai/deepseek-v4.1-flash")
DEFAULT_PROMPT = "Describe this image."

_client = None


def get_client():
    global _client
    if _client is None:
        key = os.environ.get("NVIDIA_API_KEY")
        if not key:
            raise RuntimeError("NVIDIA_API_KEY is not set")
        _client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=key)
    return _client


PAGE = """<!doctype html>
<html lang="he" dir="rtl"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NWW - ניתוח תמונה</title>
<style>body{font-family:system-ui;max-width:640px;margin:2rem auto;padding:0 1rem}
textarea,input,button{width:100%;margin:.4rem 0;padding:.6rem;font-size:1rem}
pre{white-space:pre-wrap;background:#f4f4f4;padding:1rem;border-radius:6px}</style></head>
<body><h1>ניתוח תמונה</h1><p><a href="/create">יצירת תמונה</a> · <a href="/catalog">תמונת קטלוג</a></p>
<form id="f"><input type="file" name="image" accept="image/*" required>
<textarea name="prompt" rows="3" placeholder="מה לשאול על התמונה?"></textarea>
<button>שלח</button></form><pre id="out"></pre>
<script>
document.getElementById('f').onsubmit=async e=>{
  e.preventDefault();const o=document.getElementById('out');o.textContent='...';
  const r=await fetch('/analyze',{method:'POST',body:new FormData(e.target)});
  const j=await r.json();o.textContent=j.text||j.error;};
</script></body></html>"""


GEN_URL = os.environ.get(
    "NVIDIA_IMAGE_URL",
    "https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux.2-klein-4b",
)

GEN_PAGE = """<!doctype html>
<html lang="he" dir="rtl"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NWW - יצירת תמונה</title>
<style>body{font-family:system-ui;max-width:640px;margin:2rem auto;padding:0 1rem}
textarea,input,button,select{width:100%;margin:.4rem 0;padding:.6rem;font-size:1rem}
img{max-width:100%;border-radius:6px}pre{white-space:pre-wrap}</style></head>
<body><h1>יצירת תמונה</h1><p><a href="/">ניתוח תמונה</a> · <a href="/catalog">תמונת קטלוג</a></p>
<form id="f"><textarea name="prompt" rows="4" required
 placeholder="באנגלית, כמו תיאור צילום: מה בתמונה, איפה, תאורה, עדשה. למשל: gold engraved bracelet on a woman's wrist, soft window light, macro shot"></textarea>
<select name="size"><option value="768x1344">אנכי 768x1344</option>
<option value="1024x1024">ריבוע 1024x1024</option>
<option value="1344x768">אופקי 1344x768</option></select>
<select name="model"><option value="">ללא דוגמן/ית</option>
<option value="soft" selected>על דוגמן/ית – עור טבעי ועדין</option>
<option value="real">על דוגמן/ית – עור מציאותי מאוד</option></select>
<button>צור תמונה</button></form><div id="out"></div>
<script>
document.getElementById('f').onsubmit=async e=>{
  e.preventDefault();const o=document.getElementById('out');o.textContent='...';
  const d=Object.fromEntries(new FormData(e.target));
  const r=await fetch('/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(d)});
  const j=await r.json();
  o.innerHTML=j.image?'<img src="data:image/jpeg;base64,'+j.image+'">':'<pre></pre>';
  if(!j.image)o.firstChild.textContent=j.error||JSON.stringify(j);};
</script></body></html>"""


MODEL_SUFFIX = {
    "soft": ", worn by an elegant model, natural healthy glowing skin, smooth and softly retouched, "
            "soft diffused studio light, 85mm lens, shallow depth of field, "
            "beauty and jewelry editorial photography",
    "real": ", worn by a model, natural skin with subtle fine texture, soft window light, "
            "85mm lens f/2.8, shallow depth of field, fashion editorial photography, photorealistic",
}


# FLUX understands English only; Hebrew words are ignored, so "טניס" alone gives a tennis shoe.
GLOSSARY = [
    ("צמיד טניס", "diamond tennis bracelet, a thin flexible line bracelet of evenly set small round sparkling stones, fine jewelry worn on the wrist"),
    ("צמידים", "bracelets, fine jewelry"),
    ("צמיד", "bracelet, fine jewelry worn on the wrist"),
    ("שרשרת", "necklace with a chain, fine jewelry worn around the neck"),
    ("עגילים", "earrings, fine jewelry"),
    ("עגיל", "earring, fine jewelry"),
    ("טבעת", "ring, fine jewelry worn on a finger"),
    ("חריטה", "with a delicate engraving"),
    ("זהב", "gold"),
    ("כסף", "silver"),
    ("יד", "hand"),
    ("צוואר", "neck"),
    ("דוגמנית", "female model"),
    ("דוגמן", "male model"),
]


def _expand(prompt):
    for he, en in GLOSSARY:
        prompt = re.sub(rf"(?<!\w){re.escape(he)}(?!\w)", en, prompt)
    return prompt


CAT_PAGE = """<!doctype html>
<html lang="he" dir="rtl"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NWW - תמונת קטלוג</title>
<style>body{font-family:system-ui;max-width:640px;margin:2rem auto;padding:0 1rem}
textarea,input,button,select{width:100%;margin:.4rem 0;padding:.6rem;font-size:1rem}
img{max-width:100%;border-radius:6px;border:1px solid #ddd}pre{white-space:pre-wrap}</style></head>
<body><h1>תמונת קטלוג על רקע לבן</h1><p><a href="/create">יצירת תמונה</a> · <a href="/">ניתוח תמונה</a></p>
<form id="f"><input type="file" name="image" accept="image/*" required>
<select name="size"><option value="1024x1024">ריבוע 1024x1024</option>
<option value="832x1248">אנכי 832x1248</option>
<option value="1248x832">אופקי 1248x832</option></select>
<textarea name="extra" rows="2" placeholder="בקשות נוספות (לא חובה). בלי זה לא ישתנה כלום חוץ מהרקע"></textarea>
<button>צור תמונת קטלוג</button></form><div id="out"></div>
<script>
document.getElementById('f').onsubmit=async e=>{
  e.preventDefault();const o=document.getElementById('out');o.textContent='...';
  const r=await fetch('/catalog',{method:'POST',body:new FormData(e.target)});
  const j=await r.json();
  o.innerHTML=j.image?'<img src="data:image/jpeg;base64,'+j.image+'">':'<pre></pre>';
  if(!j.image)o.firstChild.textContent=j.error||JSON.stringify(j);};
</script></body></html>"""

CATALOG_PROMPT = (
    "Edit this product photo into a clean e-commerce catalog image. "
    "Place the exact same product on a pure white seamless background (#FFFFFF), "
    "centered, with a soft subtle natural shadow beneath it, even soft studio lighting. "
    "Keep the product exactly as in the input image: same shape, proportions, colors, materials, "
    "stones, engraving, text, clasp, chain links and every small detail. "
    "Do not change, redesign, add, remove or reposition any part of the product. "
    "Do not add any props, people, hands, text, logos or decorations."
)


def _find_image_b64(body):
    if isinstance(body, dict):
        if isinstance(body.get("image"), str):
            return body["image"]
        for key in ("artifacts", "data", "images"):
            items = body.get(key)
            if isinstance(items, list) and items:
                first = items[0]
                if isinstance(first, str):
                    return first
                if isinstance(first, dict):
                    for k in ("base64", "b64_json", "image"):
                        if isinstance(first.get(k), str):
                            return first[k]
    return None


@app.get("/")
def index():
    return PAGE


@app.get("/create")
def create_page():
    return GEN_PAGE


@app.post("/generate")
def generate():
    data = request.get_json(silent=True) or {}
    prompt = (data.get("prompt") or "").strip()
    if not prompt:
        return jsonify(error="missing 'prompt'"), 400
    prompt = _expand(prompt)
    if "tennis bracelet" in prompt.lower() and "shoe" not in prompt.lower():
        prompt += ", close-up of the jewelry, no shoes, no sports equipment"
    prompt += MODEL_SUFFIX.get(data.get("model") or "", "")
    try:
        w, h = (int(x) for x in str(data.get("size", "768x1344")).split("x"))
    except ValueError:
        return jsonify(error="bad 'size', use WIDTHxHEIGHT"), 400
    key = os.environ.get("NVIDIA_API_KEY")
    if not key:
        return jsonify(error="NVIDIA_API_KEY is not set"), 500
    try:
        resp = requests.post(
            GEN_URL,
            headers={"Authorization": f"Bearer {key}", "Accept": "application/json"},
            json={"prompt": prompt, "width": w, "height": h,
                  "seed": int(data.get("seed", 0)), "steps": 4},
            timeout=110,
        )
        if resp.status_code >= 400:
            return jsonify(error=f"NVIDIA {resp.status_code}: {resp.text[:500]}"), 502
        body = resp.json()
    except Exception as exc:  # noqa: BLE001
        return jsonify(error=str(exc)), 502
    img = _find_image_b64(body)
    if not img:
        return jsonify(error="no image in response", keys=list(body)[:10]), 502
    return jsonify(image=img)


@app.get("/health")
def health():
    return jsonify(status="ok")


@app.post("/analyze")
def analyze():
    file = request.files.get("image")
    if file is None:
        return jsonify(error="missing 'image' file"), 400
    prompt = (request.form.get("prompt") or DEFAULT_PROMPT).strip()
    mime = file.mimetype if (file.mimetype or "").startswith("image/") else "image/jpeg"
    b64 = base64.b64encode(file.read()).decode()
    try:
        completion = get_client().chat.completions.create(
            model=MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                ],
            }],
            temperature=1,
            top_p=0.95,
            max_tokens=2048,
            stream=False,
        )
    except Exception as exc:  # noqa: BLE001
        return jsonify(error=str(exc)), 502
    return jsonify(text=completion.choices[0].message.content)


@app.get("/catalog")
def catalog_page():
    return CAT_PAGE


@app.post("/catalog")
def catalog():
    file = request.files.get("image")
    if file is None:
        return jsonify(error="missing 'image' file"), 400
    key = os.environ.get("NVIDIA_API_KEY")
    if not key:
        return jsonify(error="NVIDIA_API_KEY is not set"), 500
    try:
        w, h = (int(x) for x in (request.form.get("size") or "1024x1024").split("x"))
    except ValueError:
        return jsonify(error="bad 'size'"), 400
    try:
        im = Image.open(file.stream)
        im = ImageOps.exif_transpose(im)
        if im.mode in ("RGBA", "LA", "P"):
            bg = Image.new("RGB", im.size, (255, 255, 255))
            im = im.convert("RGBA")
            bg.paste(im, mask=im.split()[-1])
            im = bg
        else:
            im = im.convert("RGB")
        im.thumbnail((1024, 1024))
        buf = io.BytesIO()
        im.save(buf, "PNG")
    except Exception as exc:  # noqa: BLE001
        return jsonify(error=f"cannot read image: {exc}"), 400
    b64 = base64.b64encode(buf.getvalue()).decode()
    uri = f"data:image/png;base64,{b64}"
    prompt = CATALOG_PROMPT
    extra = (request.form.get("extra") or "").strip()
    if extra:
        prompt += " Additional request from the user, apply only this: " + _expand(extra)
    errors = []
    for img in ([uri], uri, [b64], b64):
        try:
            resp = requests.post(
                GEN_URL,
                headers={"Authorization": f"Bearer {key}", "Accept": "application/json"},
                json={"prompt": prompt, "image": img, "width": w, "height": h,
                      "seed": 0, "steps": 4},
                timeout=110,
            )
        except Exception as exc:  # noqa: BLE001
            return jsonify(error=str(exc)), 502
        if resp.status_code < 400:
            body = resp.json()
            out = _find_image_b64(body)
            if out:
                return jsonify(image=out)
            return jsonify(error="no image in response", keys=list(body)[:10]), 502
        errors.append(f"{type(img).__name__}: {resp.status_code} {resp.text[:200]}")
        if resp.status_code not in (400, 422):
            break
    last_err = "NVIDIA rejected all image formats: " + " | ".join(errors)
    return jsonify(error=last_err), 502


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
