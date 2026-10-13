import base64
import os

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
<body><h1>ניתוח תמונה</h1>
<form id="f"><input type="file" name="image" accept="image/*" required>
<textarea name="prompt" rows="3" placeholder="מה לשאול על התמונה?"></textarea>
<button>שלח</button></form><pre id="out"></pre>
<script>
document.getElementById('f').onsubmit=async e=>{
  e.preventDefault();const o=document.getElementById('out');o.textContent='...';
  const r=await fetch('/analyze',{method:'POST',body:new FormData(e.target)});
  const j=await r.json();o.textContent=j.text||j.error;};
</script></body></html>"""


@app.get("/")
def index():
    return PAGE


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
