# PoliMeme Decode — Local Inference Backend

Model/preprocessing code matches the supplied `4_Poli_Meme_Decode_fixed.ipynb`.

## Prerequisites
- 64-bit Python 3.11 (recommended), enough RAM for a 211.76M-parameter model + EasyOCR.
- Local `models/best_model.pt` (~807 MB); **keep it out of Git**.
- Internet on FIRST prediction for `roberta-base` config/tokenizer and EasyOCR weights.

## Windows PowerShell (run in project root)

```powershell
python --version
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

If `py -3.11` is unavailable, install 64-bit Python 3.11 first, or use `python -m venv .venv` if `python --version` is a compatible version.

## Health check
Open http://127.0.0.1:8000/api/health. Expected:
```json
{"status":"ok","model_file_found":true,"model_loaded":false,"device":"cpu"}
```
The model is **not loaded on health check**, only on first image prediction. The first prediction can be slow and may fetch additional weights/configs.

## Frontend
Keep the existing frontend server on port 5500 (separate terminal):
`python -m http.server 5500 --directory frontend`

The existing frontend expects `GET /api/health`, `POST /api/predict`, multipart field `file`. No JS changes required for local testing.

## Hosting note
The 807 MB checkpoint plus PyTorch + ViT + RoBERTa + EasyOCR requires much more RAM than a typical 512 MB free container. This starter is for **local development**, not guaranteed free-cloud deployment.
