import base64
import os

import requests
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
<body><h1>ניתוח תמונה</h1><p><a href="/create">יצירת תמונה</a></p>
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
<body><h1>יצירת תמונה</h1><p><a href="/">ניתוח תמונה</a></p>
<form id="f"><textarea name="prompt" rows="4" required
 placeholder="באנגלית, כמו תיאור צילום: מה בתמונה, איפה, תאורה, עדשה. למשל: gold engraved bracelet on a woman's wrist, soft window light, macro shot"></textarea>
<select name="size"><option value="768x1344">אנכי 768x1344</option>
<option value="1024x1024">ריבוע 1024x1024</option>
<option value="1344x768">אופקי 1344x768</option></select>
<label style="display:block;margin:.6rem 0"><input type="checkbox" name="model" style="width:auto;margin-inline-end:.5rem">על דוגמן/ית – עור עם טקסטורה מציאותית</label>
<button>צור תמונה</button></form><div id="out"></div>
<script>
document.getElementById('f').onsubmit=async e=>{
  e.preventDefault();const o=document.getElementById('out');o.textContent='...';
  const d=Object.fromEntries(new FormData(e.target));d.model=e.target.model.checked;
  const r=await fetch('/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(d)});
  const j=await r.json();
  o.innerHTML=j.image?'<img src="data:image/jpeg;base64,'+j.image+'">':'<pre></pre>';
  if(!j.image)o.firstChild.textContent=j.error||JSON.stringify(j);};
</script></body></html>"""


MODEL_SUFFIX = (
    ", real person wearing it, natural skin with visible pores, fine skin texture, "
    "subtle imperfections and fine hairs, soft natural window light, "
    "shot on 85mm lens f/2.8, shallow depth of field, professional fashion editorial photography, "
    "photorealistic, not retouched, not plastic"
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
    if data.get("model"):
        prompt += MODEL_SUFFIX
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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
