# AI Financial Investigation System — Backend

FastAPI backend that turns a bank-statement file (CSV / XLSX / PDF) into a
forensic investigation report. Runs Pandas extraction + OpenAI narrative.

## Endpoints

| Method | Path        | Purpose                                |
|--------|-------------|----------------------------------------|
| GET    | `/`         | Service info                           |
| GET    | `/health`   | Liveness probe                         |
| POST   | `/analyze`  | `multipart/form-data` — field `file`   |
| GET    | `/docs`     | Auto-generated Swagger UI              |

## Local run

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                # add your OPENAI_API_KEY
python main.py                      # http://localhost:8000/docs
```

## Test from terminal

```bash
curl -F "file=@statement.csv" http://localhost:8000/analyze
```

## Deploy on Render (free)

1. Push this `backend/` folder to a GitHub repo (it can be the repo root).
2. On https://render.com → **New → Web Service** → connect your repo.
3. Render auto-detects `render.yaml`. If not, set:
   - **Environment:** Python
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Add env var `OPENAI_API_KEY` in the Render dashboard.
5. Deploy → you get a public URL like
   `https://ai-financial-investigation.onrender.com`.
6. Put that URL into the Android app
   (`android-app/app/src/main/java/com/cyberforensic/app/network/RetrofitClient.kt`).

## Notes

- Without `OPENAI_API_KEY`, heuristic analysis still works; AI summary returns a
  graceful message.
- Max upload: 25 MB.
- Render free instances sleep after 15 min idle — first request after sleep
  takes ~30 s to wake.
