# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# copy files
COPY gemini_file_search_api.py /app/
COPY requirements.txt /app/

# ensure pip, then install
RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt

# create upload folder & ensure data file exists
RUN mkdir -p /app/uploads
RUN python - <<'PY'
from pathlib import Path, PurePath
p = Path("gemini_stores.json")
if not p.exists():
    p.write_text('{"file_stores": {}, "current_store_name": null}')
Path("uploads").mkdir(parents=True, exist_ok=True)
PY

EXPOSE 8000

CMD ["uvicorn", "gfsapi:app", "--host", "0.0.0.0", "--port", "8000"]
