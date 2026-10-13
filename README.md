# NWW

Small Flask service that sends an image plus a prompt to an NVIDIA-hosted
multimodal model (`deepseek-ai/deepseek-v4.1-flash`) and returns the text.

- `GET /` – simple upload form
- `POST /analyze` – multipart form: `image` (file), `prompt` (text) → `{"text": "..."}`
- `GET /health`

## Config

Set `NVIDIA_API_KEY` as an environment variable (never commit it).
Optional: `NVIDIA_MODEL`.

## Run

    pip install -r requirements.txt
    gunicorn app:app
